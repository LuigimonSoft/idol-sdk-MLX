"""Synthetic octree checkpoint and forward parity against the pinned SDK."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("mlx") and os.environ.get("IDOL_REFERENCE_SOURCE"),
                     "MLX and pinned reference source are required")
class OctreeParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        import numpy as np
        import torch
        from mlx.utils import tree_flatten
        from safetensors.torch import save_file
        from idol_mlx.octree import OctreeProbabilityFixedlenDecoder
        from idol_mlx.octree_weights import load_octree_checkpoint
        from idol_mlx.weights import CheckpointError

        sys.path.insert(0, str(Path(os.environ["IDOL_REFERENCE_SOURCE"]).resolve()))
        from idol.models.decoder import OctreeProbabilityFixedlenDecoder as TorchDecoder

        cls.mx, cls.np, cls.torch = mx, np, torch
        cls.torch_decoder, cls.mlx_decoder = TorchDecoder, OctreeProbabilityFixedlenDecoder
        cls.load_checkpoint, cls.save_file = staticmethod(load_octree_checkpoint), staticmethod(save_file)
        cls.tree_flatten, cls.checkpoint_error = staticmethod(tree_flatten), CheckpointError
        device = os.environ.get("IDOL_TEST_MLX_DEVICE", "cpu")
        if device not in ("cpu", "gpu"):
            raise ValueError("IDOL_TEST_MLX_DEVICE must be cpu or gpu")
        cls.previous_device = mx.default_device()
        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
        cls.addClassCleanup(mx.set_default_device, cls.previous_device)

    def make_models(self):
        args = dict(model_channels=24, cond_channels=8, num_blocks=2, num_heads=2, mlp_ratio=1.5)
        self.torch.manual_seed(10)
        reference = self.torch_decoder(**args).eval().float()
        reference.dtype = self.torch.float32
        return reference, self.mlx_decoder(**args)

    def test_logits_probabilities_and_conditioning(self):
        mx, np, torch = self.mx, self.np, self.torch
        reference, model = self.make_models()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "octree.safetensors"
            self.save_file({key: value.detach().contiguous() for key, value in reference.state_dict().items()},
                           str(path))
            self.load_checkpoint(model, path)
            rng = np.random.default_rng(42)
            for batch, length, context_len in ((1, 3, 4), (2, 5, 2)):
                with self.subTest(batch=batch, length=length, context=context_len):
                    xyz = rng.random((batch, length, 3), dtype=np.float32)
                    level = np.arange(batch, dtype=np.float32) + 4
                    context = rng.standard_normal((batch, context_len, 8)).astype(np.float32)
                    with torch.no_grad():
                        expected = reference(torch.from_numpy(xyz), torch.from_numpy(level), torch.from_numpy(context))
                        expected_probs = torch.softmax(expected, dim=-1).numpy()
                        changed = reference(torch.from_numpy(xyz), torch.from_numpy(level),
                                            torch.zeros_like(torch.from_numpy(context)))
                    actual = model(mx.array(xyz), mx.array(level), mx.array(context))
                    probs = mx.softmax(actual, axis=-1)
                    mx.eval(actual, probs)
                    self.assertEqual(actual.shape, (batch, length, 8))
                    np.testing.assert_allclose(np.array(actual), expected.numpy(), atol=2e-5, rtol=2e-4)
                    np.testing.assert_allclose(np.array(probs), expected_probs, atol=2e-5, rtol=2e-4)
                    np.testing.assert_allclose(np.array(probs).sum(axis=-1), 1, atol=1e-6)
                    self.assertGreater(float(torch.max(torch.abs(expected - changed))), 1e-5)

    def test_invalid_checkpoint_does_not_mutate_model(self):
        reference, model = self.make_models()
        source = {key: value.detach().contiguous() for key, value in reference.state_dict().items()}
        original = {key: self.np.array(value) for key, value in self.tree_flatten(model.parameters())}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.safetensors"
            for kind in ("missing", "unexpected", "shape", "dtype"):
                with self.subTest(kind=kind):
                    tensors = dict(source)
                    if kind == "missing":
                        tensors.pop("in_proj.bias")
                    elif kind == "unexpected":
                        tensors["unexpected"] = self.torch.zeros(1)
                    elif kind == "shape":
                        tensors["in_proj.bias"] = self.torch.zeros(1)
                    else:
                        tensors["in_proj.bias"] = tensors["in_proj.bias"].half()
                    self.save_file(tensors, str(path))
                    with self.assertRaises(self.checkpoint_error):
                        self.load_checkpoint(model, path)
                    for name, value in self.tree_flatten(model.parameters()):
                        self.np.testing.assert_array_equal(self.np.array(value), original[name])

    def test_checkpoint_diagnostic_with_synthetic_weights(self):
        import json
        from scripts.validate_octree_checkpoint import compare_checkpoint

        reference, _ = self.make_models()
        config = {"denoiser": {"name": "unused", "args": {}},
                  "decoder": {"name": "OctreeProbabilityFixedlenDecoder",
                              "args": {"model_channels": 24, "cond_channels": 8, "num_blocks": 2,
                                       "num_heads": 2, "mlp_ratio": 1.5}},
                  "decoder_gs": {"name": "unused", "args": {}}, "image_size": 16, "max_voxel_level": 1,
                  "image_encoders": {"dinov3": "unused", "flux2_vae": "unused"}}
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            path = Path(directory) / "octree.safetensors"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            self.save_file({key: value.detach().contiguous() for key, value in reference.state_dict().items()},
                           str(path))
            report = compare_checkpoint(config_path, path, cells=3, context_tokens=2, level=4)
            self.assertEqual(report["comparisons"]["logits"]["count"], 24)
            self.assertTrue(all(value["passed"] for value in report["comparisons"].values()), report)
            for name in ("cells", "context_tokens", "level", "seed"):
                with self.subTest(argument=name), self.assertRaisesRegex(ValueError, name):
                    compare_checkpoint(config_path, path, **{name: -1})

    def test_profile_synthetic_checkpoint_and_validate_inputs(self):
        import json
        from scripts.profile_octree import profile_checkpoint

        reference, _ = self.make_models()
        args = {"model_channels": 24, "cond_channels": 8, "num_blocks": 2,
                "num_heads": 2, "mlp_ratio": 1.5}
        config = {"denoiser": {"name": "unused", "args": {}},
                  "decoder": {"name": "OctreeProbabilityFixedlenDecoder", "args": args},
                  "decoder_gs": {"name": "unused", "args": {}}, "image_size": 16, "max_voxel_level": 1,
                  "image_encoders": {"dinov3": "unused", "flux2_vae": "unused"}}
        with tempfile.TemporaryDirectory() as directory:
            config_path, checkpoint = Path(directory) / "config.json", Path(directory) / "octree.safetensors"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            self.save_file({name: value.detach().contiguous() for name, value in reference.state_dict().items()},
                           str(checkpoint))
            for points, level in ((0, 0), (8, 2)):
                with self.subTest(num_points=points):
                    result = profile_checkpoint(config_path, checkpoint, num_points=points, level=level,
                                                context_tokens=2, seed=11)
                    self.assertTrue(all(result[key] for key in ("shape_ok", "finite", "in_bounds")), result)
                    self.assertEqual(result["num_points"], points)
                    self.assertEqual(result["level"], level)
                    self.assertGreaterEqual(result["seconds"], 0)
                    self.assertGreater(result["mlx_baseline_bytes"], 0)
                    self.assertGreaterEqual(result["mlx_peak_bytes"], result["mlx_baseline_bytes"])
                    self.assertGreater(result["process_peak_rss_bytes"], 0)
            for name in ("num_points", "level", "context_tokens", "seed"):
                invalid = dict(num_points=8, level=2, context_tokens=2, seed=11)
                invalid[name] = -1
                with self.subTest(argument=name), self.assertRaisesRegex(ValueError, name):
                    profile_checkpoint(config_path, checkpoint, **invalid)

    def test_rejects_non_fp32_target_before_loading(self):
        _, model = self.make_models()
        model.in_proj.weight = model.in_proj.weight.astype(self.mx.float16)
        with self.assertRaisesRegex(ValueError, "target model tensors must be float32"):
            self.load_checkpoint(model, "unused.safetensors")
