"""Controlled Gaussian attribute comparisons with the pinned PyTorch SDK."""

import importlib.util
import os
from pathlib import Path
import sys
import unittest


@unittest.skipUnless(importlib.util.find_spec("mlx") and os.environ.get("IDOL_REFERENCE_SOURCE"),
                     "MLX and pinned reference source are required")
class GaussianAttributeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        import numpy as np
        import torch
        from idol_mlx.gaussian_attributes import decode_gaussian_attributes

        sys.path.insert(0, str(Path(os.environ["IDOL_REFERENCE_SOURCE"]).resolve()))
        from idol.models.decoder import GaussianFixedlenDecoder
        cls.mx, cls.np, cls.torch = mx, np, torch
        cls.decode, cls.reference = staticmethod(decode_gaussian_attributes), GaussianFixedlenDecoder
        device = os.environ.get("IDOL_TEST_MLX_DEVICE", "cpu")
        if device not in ("cpu", "gpu"):
            raise ValueError("IDOL_TEST_MLX_DEVICE must be cpu or gpu")
        previous = mx.default_device()
        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
        cls.addClassCleanup(mx.set_default_device, previous)

    def config(self, per_anchor=3):
        return {"num_gaussians": per_anchor,
                "lr": {"_xyz": 0.7, "_features_dc": 1.3, "_scaling": 0.4, "_rotation": 0.1, "_opacity": 1.2},
                "perturbe_size": 1.5, "scaling_bias": 0.004, "opacity_bias": 0.1,
                "3d_filter_kernel_size": 0.0009}

    def compare_with_reference(self, points, features, perturbation, base, config):
        mx, np, torch = self.mx, self.np, self.torch
        ref = self.reference(in_channels=3, model_channels=24, cond_channels=8, num_blocks=1, num_heads=2,
                             representation_config=config, mlp_ratio=1.5)
        with torch.no_grad():
            ref.points_offset_perturbation.copy_(torch.from_numpy(perturbation))
            ref.base_offset_scale.copy_(torch.tensor(base, dtype=torch.float32))
            expected = ref.to_representation(torch.from_numpy(points), torch.from_numpy(features))
        result = self.decode(mx.array(points), mx.array(features), representation_config=config,
                             points_offset_perturbation=mx.array(perturbation), base_offset_scale=mx.array(base))
        mx.eval(result)
        self.assertEqual(set(result), {"xyz", "features_dc", "scaling", "rotation", "opacity"})
        for key in result:
            expected_array = np.stack([getattr(item, key).detach().numpy() for item in expected])
            actual = np.array(result[key])
            self.assertEqual(actual.shape, expected_array.shape)
            np.testing.assert_allclose(actual, expected_array, atol=2e-5, rtol=2e-4, err_msg=key)
        return result

    def test_multiple_batches_and_shapes_match_reference(self):
        np = self.np
        rng = np.random.default_rng(12)
        for batch, anchors, per_anchor in ((1, 1, 1), (2, 4, 3), (1, 7, 32)):
            with self.subTest(batch=batch, anchors=anchors, per_anchor=per_anchor):
                points = rng.random((batch, anchors, 3), dtype=np.float32)
                features = rng.standard_normal((batch, anchors, per_anchor * 15)).astype(np.float32)
                perturbation = rng.standard_normal((per_anchor, 3)).astype(np.float32) * 0.05
                result = self.compare_with_reference(points, features, perturbation, np.float32(-0.3),
                                                     self.config(per_anchor))
                self.assertEqual(result["xyz"].shape, (batch, anchors * per_anchor, 3))
                self.assertEqual(result["features_dc"].shape, (batch, anchors * per_anchor, 1, 3))
                opacity = np.array(result["opacity"])
                self.assertTrue(np.all((opacity >= 0) & (opacity <= 1)))
                self.assertTrue(np.all(np.array(result["scaling"]) >= 0.0009))
                np.testing.assert_allclose(np.linalg.norm(np.array(result["rotation"]), axis=-1), 1, atol=2e-6)

    def test_zero_quaternion_saturation_and_sh_color(self):
        np = self.np
        config = self.config(1)
        features = np.zeros((1, 2, 15), dtype=np.float32)
        features[0, 0, 3:6] = [-2, 3, 4]
        features[0, 0, 9:13] = [-10, 0, 0, 0]  # rotation lr=0.1 cancels identity
        features[0, :, 13] = [-100, 100]
        result = self.compare_with_reference(np.full((1, 2, 3), 0.5, dtype=np.float32), features,
                                             np.zeros((1, 3), dtype=np.float32), np.float32(0), config)
        np.testing.assert_array_equal(np.array(result["rotation"])[0, 0], [0, 0, 0, 0])
        self.assertLess(float(np.array(result["features_dc"])[0, 0, 0, 0]), 0)
        self.assertLess(float(np.array(result["opacity"])[0, 0, 0]), 1e-6)
        self.assertGreater(float(np.array(result["opacity"])[0, 1, 0]), 1 - 1e-6)

    def test_synthetic_network_and_conversion_together(self):
        import tempfile
        from safetensors.torch import save_file
        from idol_mlx.gaussian_decoder import GaussianFixedlenDecoder
        from idol_mlx.gaussian_weights import load_gaussian_checkpoint

        mx, np, torch = self.mx, self.np, self.torch
        config = self.config(3)
        args = dict(in_channels=3, model_channels=24, cond_channels=8, num_blocks=2, num_heads=2,
                    representation_config=config, mlp_ratio=1.5)
        torch.manual_seed(31)
        reference = self.reference(**args).eval().float()
        reference.dtype = torch.float32
        with torch.no_grad():
            reference.points_offset_perturbation.copy_(torch.tensor([[0.1, -0.2, 0.3], [0.2, 0.1, -0.1],
                                                                    [-0.3, 0.2, 0.0]]))
            reference.base_offset_scale.fill_(-0.2)
        candidate = GaussianFixedlenDecoder(**args)
        rng = np.random.default_rng(18)
        points = rng.random((2, 5, 3), dtype=np.float32)
        condition = rng.standard_normal((2, 4, 8)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "gaussian.safetensors"
            save_file({key: value.detach().contiguous() for key, value in reference.state_dict().items()},
                      str(checkpoint))
            load_gaussian_checkpoint(candidate, checkpoint)
            with torch.no_grad():
                features = reference(torch.from_numpy(points), torch.from_numpy(condition))
                expected = reference.to_representation(torch.from_numpy(points), features)
            actual_features = candidate(mx.array(points), mx.array(condition))
            actual = self.decode(mx.array(points), actual_features, representation_config=config,
                                 points_offset_perturbation=candidate.points_offset_perturbation,
                                 base_offset_scale=candidate.base_offset_scale)
            mx.eval(actual)
            for name, value in actual.items():
                expected_value = np.stack([getattr(item, name).detach().numpy() for item in expected])
                np.testing.assert_allclose(np.array(value), expected_value, atol=2e-5, rtol=2e-4, err_msg=name)

    def test_trained_checkpoint_diagnostic_with_synthetic_weights(self):
        import json
        import tempfile
        from safetensors.torch import save_file
        from scripts.validate_gaussian_attributes import compare_checkpoint

        config = self.config(3)
        args = dict(in_channels=3, model_channels=24, cond_channels=8, num_blocks=1, num_heads=2,
                    representation_config=config, mlp_ratio=1.5)
        self.torch.manual_seed(15)
        reference = self.reference(**args).eval().float()
        data = {"denoiser": {"name": "unused", "args": {}}, "decoder": {"name": "unused", "args": {}},
                "decoder_gs": {"name": "GaussianFixedlenDecoder", "args": args},
                "image_size": 16, "max_voxel_level": 1,
                "image_encoders": {"dinov3": "unused", "flux2_vae": "unused"}}
        with tempfile.TemporaryDirectory() as directory:
            config_path, checkpoint = Path(directory) / "config.json", Path(directory) / "gaussian.safetensors"
            config_path.write_text(json.dumps(data), encoding="utf-8")
            save_file({name: value.detach().contiguous() for name, value in reference.state_dict().items()},
                      str(checkpoint))
            report = compare_checkpoint(config_path, checkpoint, anchors=3, context_tokens=2)
            self.assertTrue(report["shared_reference_features"])
            self.assertEqual(set(report["comparisons"]), {"xyz", "features_dc", "scaling", "rotation", "opacity"})
            self.assertTrue(all(item["passed"] for item in report["comparisons"].values()), report)
            self.assertEqual(report["comparisons"]["xyz"]["count"], 27)
            for name in ("anchors", "context_tokens", "seed"):
                with self.subTest(argument=name), self.assertRaisesRegex(ValueError, name):
                    compare_checkpoint(config_path, checkpoint, **{name: -1})

    def test_invalid_shapes_types_and_configuration(self):
        mx = self.mx
        config = self.config(2)
        points, features = mx.zeros((1, 2, 3)), mx.zeros((1, 2, 30))
        perturbation, base = mx.zeros((2, 3)), mx.zeros(())
        def call(**overrides):
            kwargs = dict(points=points, features=features, representation_config=config,
                          points_offset_perturbation=perturbation, base_offset_scale=base)
            kwargs.update(overrides)
            return self.decode(**kwargs)
        for kwargs in ({"points": mx.zeros((1, 0, 3))}, {"features": mx.zeros((1, 2, 29))},
                       {"points_offset_perturbation": mx.zeros((1, 3))}, {"base_offset_scale": mx.zeros((1,))},
                       {"features": features.astype(mx.float16)}):
            with self.subTest(keys=list(kwargs)), self.assertRaises(ValueError):
                call(**kwargs)
        for key, value in (("num_gaussians", True), ("scaling_bias", 0), ("opacity_bias", 1),
                           ("3d_filter_kernel_size", -1), ("perturbe_size", float("nan")), ("lr", {})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                call(representation_config={**config, key: value})
