"""Paired Euler sampling checks against the pinned PyTorch reference."""

import importlib.util
import os
from pathlib import Path
import sys
import unittest


@unittest.skipUnless(importlib.util.find_spec("mlx") and os.environ.get("IDOL_REFERENCE_SOURCE"),
                     "MLX and pinned reference source are required")
class SamplerParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        import numpy as np
        import torch
        from idol_mlx.sampler import sample_euler, time_grid

        sys.path.insert(0, str(Path(os.environ["IDOL_REFERENCE_SOURCE"]).resolve()))
        from idol.sampler import sample_flow

        cls.mx, cls.np, cls.torch = mx, np, torch
        cls.mlx_sample, cls.time_grid = staticmethod(sample_euler), staticmethod(time_grid)
        cls.torch_sample = staticmethod(sample_flow)
        cls.previous_device = mx.default_device()
        device = os.environ.get("IDOL_TEST_MLX_DEVICE", "cpu")
        if device not in ("cpu", "gpu"):
            raise ValueError("IDOL_TEST_MLX_DEVICE must be cpu or gpu")
        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)

    @classmethod
    def tearDownClass(cls):
        cls.mx.set_default_device(cls.previous_device)

    def test_grid_matches_reference_and_invalid_values_fail(self):
        np = self.np
        for steps, shift in ((1, 1.0), (5, 3.0), (7, 0.5)):
            raw = np.linspace(1, 0, steps + 1)
            expected = shift * raw / (1 + (shift - 1) * raw)
            np.testing.assert_allclose(self.time_grid(steps, shift), expected, atol=1e-15, rtol=1e-15)
        for steps, shift in ((0, 1), (-1, 1), (True, 1), (1, 0), (1, float("nan")), (1, float("inf"))):
            with self.subTest(steps=steps, shift=shift), self.assertRaises(ValueError):
                self.time_grid(steps, shift)

    def test_tiny_positive_shift_preserves_endpoints_and_full_integration(self):
        mx, np = self.mx, self.np
        for shift in (1e-16, 1e-20):
            with self.subTest(shift=shift):
                self.assertEqual(self.time_grid(1, shift), (1.0, 0.0))
                grid = self.time_grid(4, shift)
                self.assertEqual((grid[0], grid[-1]), (1.0, 0.0))
                self.assertTrue(all(a >= b for a, b in zip(grid, grid[1:])))
                result = self.mlx_sample(lambda sample, time, condition: condition,
                                         mx.array([[10.0]]), mx.array([[2.0]]), steps=1, rescale_t=shift)
                mx.eval(result)
                np.testing.assert_array_equal(np.array(result), np.array([[8.0]], dtype=np.float32))

    def test_known_velocity_and_time_dependent_trajectories(self):
        mx, np, torch = self.mx, self.np, self.torch
        noise = np.array([[10., -2.], [3., 7.]], dtype=np.float32)
        cond = np.array([[2., -1.], [1.5, 0.25]], dtype=np.float32)
        for steps, shift, dynamic in ((1, 1.0, False), (5, 3.0, False), (7, 0.5, True), (6, 3.0, True)):
            with self.subTest(steps=steps, shift=shift, dynamic=dynamic):
                torch_times, mlx_times = [], []
                def torch_model(sample, time, condition):
                    torch_times.append(time.detach().numpy().copy())
                    return condition if not dynamic else condition + 0.2 * sample + time[:, None] * 0.001
                def mlx_model(sample, time, condition):
                    mlx_times.append(time)
                    return condition if not dynamic else condition + 0.2 * sample + time[:, None] * 0.001
                expected = self.torch_sample(torch_model, torch.from_numpy(noise), torch.from_numpy(cond),
                                             torch.from_numpy(cond), steps=steps, rescale_t=shift,
                                             cfg_strength=0, verbose=False).numpy()
                actual = self.mlx_sample(mlx_model, mx.array(noise), mx.array(cond), steps=steps, rescale_t=shift)
                mx.eval(actual, *mlx_times)
                self.assertEqual(len(mlx_times), steps)
                self.assertEqual(actual.dtype, mx.float32)
                np.testing.assert_allclose(np.array(actual), expected, atol=1e-5, rtol=1e-4)
                np.testing.assert_allclose(np.stack([np.array(t) for t in mlx_times]),
                                           np.stack(torch_times), atol=1e-4, rtol=1e-6)
                if not dynamic:
                    np.testing.assert_allclose(np.array(actual), noise - cond, atol=1e-5, rtol=1e-5)
