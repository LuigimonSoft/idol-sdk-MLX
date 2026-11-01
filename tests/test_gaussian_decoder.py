"""Synthetic Gaussian decoder and strict-checkpoint parity tests."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("mlx") and os.environ.get("IDOL_REFERENCE_SOURCE"),
                     "MLX and pinned reference source are required")
class GaussianDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        import numpy as np
        import torch
        from mlx.utils import tree_flatten
        from safetensors.torch import save_file
        from idol_mlx.gaussian_decoder import GaussianFixedlenDecoder
        from idol_mlx.gaussian_weights import load_gaussian_checkpoint
        from idol_mlx.weights import CheckpointError

        sys.path.insert(0, str(Path(os.environ["IDOL_REFERENCE_SOURCE"]).resolve()))
        from idol.models.decoder import GaussianFixedlenDecoder as TorchDecoder
        cls.mx, cls.np, cls.torch = mx, np, torch
        cls.mlx_decoder, cls.torch_decoder = GaussianFixedlenDecoder, TorchDecoder
        cls.load_checkpoint, cls.save_file = staticmethod(load_gaussian_checkpoint), staticmethod(save_file)
        cls.tree_flatten, cls.checkpoint_error = staticmethod(tree_flatten), CheckpointError
        device = os.environ.get("IDOL_TEST_MLX_DEVICE", "cpu")
        if device not in ("cpu", "gpu"):
            raise ValueError("IDOL_TEST_MLX_DEVICE must be cpu or gpu")
        previous = mx.default_device()
        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
        cls.addClassCleanup(mx.set_default_device, previous)

    def make_models(self):
        args = dict(in_channels=3, model_channels=24, cond_channels=8, num_blocks=2, num_heads=2,
                    representation_config={"num_gaussians": 3}, mlp_ratio=1.5)
        self.torch.manual_seed(13)
        reference = self.torch_decoder(**args).eval().float()
        reference.dtype = self.torch.float32
        with self.torch.no_grad():
            reference.points_offset_perturbation.fill_(0.25)
            reference.base_offset_scale.fill_(0.75)
        return reference, self.mlx_decoder(**args)

    def test_raw_features_and_buffers_match_reference(self):
        mx, np, torch = self.mx, self.np, self.torch
        ref, model = self.make_models()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gaussian.safetensors"
            self.save_file({name: value.detach().contiguous() for name, value in ref.state_dict().items()}, str(path))
            self.load_checkpoint(model, path)
            np.testing.assert_array_equal(np.array(model.points_offset_perturbation),
                                          ref.points_offset_perturbation.numpy())
            np.testing.assert_array_equal(np.array(model.base_offset_scale), ref.base_offset_scale.numpy())
            rng = np.random.default_rng(29)
            for batch, anchors, context in ((1, 3, 4), (2, 5, 2)):
                with self.subTest(batch=batch, anchors=anchors, context=context):
                    points = rng.random((batch, anchors, 3), dtype=np.float32)
                    cond = rng.standard_normal((batch, context, 8)).astype(np.float32)
                    with torch.no_grad():
                        expected = ref(torch.from_numpy(points), torch.from_numpy(cond)).numpy()
                        zero_cond = ref(torch.from_numpy(points), torch.zeros_like(torch.from_numpy(cond))).numpy()
                    actual = model(mx.array(points), mx.array(cond))
                    mx.eval(actual)
                    self.assertEqual(actual.shape, (batch, anchors, 45))
                    np.testing.assert_allclose(np.array(actual), expected, atol=2e-5, rtol=2e-4)
                    self.assertGreater(float(np.max(np.abs(expected - zero_cond))), 1e-5)

    def test_invalid_checkpoint_does_not_mutate_model(self):
        ref, model = self.make_models()
        source = {key: value.detach().contiguous() for key, value in ref.state_dict().items()}
        before = {name: self.np.array(value) for name, value in self.tree_flatten(model.parameters())}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.safetensors"
            for kind in ("missing_buffer", "unexpected", "shape", "dtype"):
                with self.subTest(kind=kind):
                    tensors = dict(source)
                    if kind == "missing_buffer":
                        tensors.pop("base_offset_scale")
                    elif kind == "unexpected":
                        tensors["unexpected"] = self.torch.zeros(1)
                    elif kind == "shape":
                        tensors["points_offset_perturbation"] = self.torch.zeros((1, 3))
                    else:
                        tensors["base_offset_scale"] = tensors["base_offset_scale"].half()
                    self.save_file(tensors, str(path))
                    with self.assertRaises(self.checkpoint_error):
                        self.load_checkpoint(model, path)
                    for name, value in self.tree_flatten(model.parameters()):
                        self.np.testing.assert_array_equal(self.np.array(value), before[name])

    def test_checkpoint_diagnostic_with_synthetic_weights(self):
        import json
        from unittest.mock import patch
        from scripts.validate_gaussian_checkpoint import compare_checkpoint, main

        reference, _ = self.make_models()
        args = dict(in_channels=3, model_channels=24, cond_channels=8, num_blocks=2, num_heads=2,
                    representation_config={"num_gaussians": 3}, mlp_ratio=1.5)
        config = {"denoiser": {"name": "unused", "args": {}}, "decoder": {"name": "unused", "args": {}},
                  "decoder_gs": {"name": "GaussianFixedlenDecoder", "args": args},
                  "image_size": 16, "max_voxel_level": 1,
                  "image_encoders": {"dinov3": "unused", "flux2_vae": "unused"}}
        with tempfile.TemporaryDirectory() as directory:
            config_path, path = Path(directory) / "config.json", Path(directory) / "gaussian.safetensors"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            self.save_file({name: value.detach().contiguous() for name, value in reference.state_dict().items()},
                           str(path))
            report = compare_checkpoint(config_path, path, anchors=3, context_tokens=2)
            self.assertEqual(report["comparisons"]["features"]["count"], 135)
            self.assertTrue(all(item["passed"] for item in report["comparisons"].values()), report)
            traced = compare_checkpoint(config_path, path, anchors=3, context_tokens=2,
                                        trace_blocks=True, compare_attention=True)
            self.assertEqual(traced["comparisons"], report["comparisons"])
            self.assertEqual([item["index"] for item in traced["block_diagnostics"]], [0, 1])
            self.assertTrue(all(item["passed"] for item in traced["block_diagnostics"]), traced)
            self.assertTrue(traced["explicit_attention_diagnostic"]["passed"], traced)
            self.assertEqual(traced["explicit_attention_diagnostic"]["count"], 135)
            failed_base = {**traced, "comparisons": {**traced["comparisons"],
                           "features": {**traced["comparisons"]["features"], "passed": False}}}
            with patch("scripts.validate_gaussian_checkpoint.compare_checkpoint", return_value=failed_base) as run, \
                    patch("scripts.validate_gaussian_checkpoint.mx.set_default_device"), patch("builtins.print"):
                self.assertEqual(main(["--config", str(config_path), "--checkpoint", str(path),
                                       "--trace-blocks", "--compare-attention"]), 1)
                self.assertTrue(run.call_args.kwargs["trace_blocks"])
                self.assertTrue(run.call_args.kwargs["compare_attention"])
            with self.assertRaisesRegex(ValueError, "at most 256"):
                compare_checkpoint(config_path, path, anchors=257, compare_attention=True)
            with self.assertRaisesRegex(ValueError, "at most 256"):
                compare_checkpoint(config_path, path, context_tokens=257, trace_blocks=True)
            with self.assertRaisesRegex(ValueError, "diagnostic flags"):
                compare_checkpoint(config_path, path, trace_blocks=1)
            for name in ("anchors", "context_tokens", "seed"):
                with self.subTest(argument=name), self.assertRaisesRegex(ValueError, name):
                    compare_checkpoint(config_path, path, **{name: -1})

    def test_rejects_invalid_configuration_and_target_dtype(self):
        _, model = self.make_models()
        model.in_proj.weight = model.in_proj.weight.astype(self.mx.float16)
        with self.assertRaisesRegex(ValueError, "target model tensors must be float32"):
            self.load_checkpoint(model, "unused.safetensors")
        args = dict(in_channels=3, model_channels=24, cond_channels=8, num_blocks=1, num_heads=2,
                    representation_config={"num_gaussians": 1})
        for value in (0, -1, True, 1.5, None):
            with self.subTest(num_gaussians=value), self.assertRaisesRegex(ValueError, "num_gaussians"):
                self.mlx_decoder(**{**args, "representation_config": {"num_gaussians": value}})
        with self.assertRaisesRegex(ValueError, "in_channels"):
            self.mlx_decoder(**{**args, "in_channels": 2})
