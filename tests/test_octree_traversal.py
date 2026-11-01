"""Controlled octree descent checks against the pinned upstream sampler."""

import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("mlx") and os.environ.get("IDOL_REFERENCE_SOURCE"),
                     "MLX and pinned reference source are required")
class OctreeTraversalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        import numpy as np
        import torch
        from idol_mlx.octree_sampling import sample_octree

        sys.path.insert(0, str(Path(os.environ["IDOL_REFERENCE_SOURCE"]).resolve()))
        from idol.models.decoder import OctreeProbabilityFixedlenDecoder
        cls.mx, cls.np, cls.torch = mx, np, torch
        cls.sample = staticmethod(sample_octree)
        cls.reference = staticmethod(OctreeProbabilityFixedlenDecoder.sample)
        device = os.environ.get("IDOL_TEST_MLX_DEVICE", "cpu")
        if device not in ("cpu", "gpu"):
            raise ValueError("IDOL_TEST_MLX_DEVICE must be cpu or gpu")
        previous = mx.default_device()
        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
        cls.addClassCleanup(mx.set_default_device, previous)

    def test_reference_matches_controlled_offsets_and_jitter(self):
        mx, np, torch = self.mx, self.np, self.torch
        features = np.array([[[4., 3., 2., 1., 0., -1., -2., -3.]],
                             [[0., 0., 0., 0., 0., 0., 0., 0.]]], dtype=np.float32)
        def torch_model(x, resolution, cond):
            self.assertEqual(tuple(resolution.shape), (2,))
            return cond[:, :1, :].expand(-1, x.shape[1], -1)
        def mlx_model(x, resolution, cond):
            self.assertEqual(tuple(resolution.shape), (2,))
            return mx.broadcast_to(cond[:, :1, :], (x.shape[0], x.shape[1], 8))
        def torch_rand(*shape, device=None):
            return torch.full(shape, 0.375, dtype=torch.float32, device=device)
        def torch_jitter(value, dtype=None):
            return torch.full_like(value, 0.375, dtype=dtype)
        def mlx_uniform(low=0, high=1, shape=(), dtype=None, key=None, stream=None):
            return mx.full(shape, 0.375, dtype=dtype)
        for levels in (0, 1, 2, 3):
            with self.subTest(level=levels), patch("torch.rand", side_effect=torch_rand), \
                    patch("torch.rand_like", side_effect=torch_jitter), \
                    patch.object(mx.random, "uniform", side_effect=mlx_uniform):
                expected = self.reference(torch_model, torch.from_numpy(features), 17, levels).numpy()
                result = self.sample(mlx_model, mx.array(features), num_points=17, level=levels, seed=99)
                mx.eval(result)
                np.testing.assert_array_equal(np.array(result), expected)
                self.assertEqual(result.shape, (2, 17, 3))

    def test_seed_reproducibility_bounds_and_empty(self):
        mx, np = self.mx, self.np
        def model(x, resolution, cond):
            return mx.zeros((*x.shape[:-1], 8))
        cond = mx.ones((2, 3, 8))
        a = self.sample(model, cond, num_points=101, level=3, seed=17)
        b = self.sample(model, cond, num_points=101, level=3, seed=17)
        c = self.sample(model, cond, num_points=101, level=3, seed=18)
        mx.eval(a, b, c)
        av, bv, cv = np.array(a), np.array(b), np.array(c)
        np.testing.assert_array_equal(av, bv)
        self.assertFalse(np.array_equal(av, cv))
        self.assertTrue(np.all((av >= 0) & (av < 1)))
        self.assertEqual(self.sample(model, cond, num_points=0, level=3, seed=17).shape, (2, 0, 3))

    def test_uniform_root_distributes_points_across_octants(self):
        mx, np = self.mx, self.np
        def model(x, resolution, cond):
            return mx.zeros((*x.shape[:-1], 8))
        result = self.sample(model, mx.ones((1, 1, 8)), num_points=4096, level=1, seed=7)
        mx.eval(result)
        cells = np.floor(np.array(result[0]) * 2).astype(np.int64)
        indices = cells[:, 0] + 2 * cells[:, 1] + 4 * cells[:, 2]
        np.testing.assert_array_equal(np.bincount(indices, minlength=8), np.full(8, 512))

    def test_trained_checkpoint_diagnostic_with_synthetic_weights(self):
        import json
        import tempfile
        from safetensors.torch import save_file
        from scripts.validate_octree_traversal import compare_checkpoint
        from idol.models.decoder import OctreeProbabilityFixedlenDecoder

        args = {"model_channels": 24, "cond_channels": 8, "num_blocks": 1,
                "num_heads": 2, "mlp_ratio": 1.5}
        self.torch.manual_seed(11)
        reference = OctreeProbabilityFixedlenDecoder(**args).eval().float()
        config = {"denoiser": {"name": "unused", "args": {}},
                  "decoder": {"name": "OctreeProbabilityFixedlenDecoder", "args": args},
                  "decoder_gs": {"name": "unused", "args": {}}, "image_size": 16, "max_voxel_level": 1,
                  "image_encoders": {"dinov3": "unused", "flux2_vae": "unused"}}
        with tempfile.TemporaryDirectory() as directory:
            config_path, path = Path(directory) / "config.json", Path(directory) / "octree.safetensors"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            save_file({name: value.detach().contiguous() for name, value in reference.state_dict().items()}, str(path))
            result = compare_checkpoint(config_path, path, num_points=8, level=2, context_tokens=2)
            self.assertTrue(result["coordinates"]["passed"], result)
            self.assertEqual(result["coordinates"]["count"], 24)
            for name in ("num_points", "level", "context_tokens", "seed"):
                with self.subTest(argument=name), self.assertRaisesRegex(ValueError, name):
                    compare_checkpoint(config_path, path, **{name: -1})

    def test_rejects_invalid_arguments_and_model_shape(self):
        mx = self.mx
        cond = mx.ones((1, 1, 8))
        def model(x, resolution, condition):
            return mx.zeros((*x.shape[:-1], 8))
        for name, values in (("num_points", (-1, 2**24, True)), ("level", (-1, 24, True)),
                             ("seed", (-1, 2**64, True))):
            for value in values:
                args = dict(num_points=4, level=1, seed=0)
                args[name] = value
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    self.sample(model, cond, **args)
        with self.assertRaisesRegex(ValueError, "cond"):
            self.sample(model, mx.ones((1, 8)), num_points=4, level=1, seed=0)
        with self.assertRaisesRegex(ValueError, "float32"):
            self.sample(model, cond.astype(mx.float16), num_points=4, level=1, seed=0)
        with self.assertRaisesRegex(ValueError, "model must return"):
            self.sample(lambda x, t, c: mx.zeros((1, 1, 7)), cond, num_points=4, level=1, seed=0)
