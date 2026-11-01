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
        from idol_mlx.sampler import sample_euler, seeded_noise, time_grid

        sys.path.insert(0, str(Path(os.environ["IDOL_REFERENCE_SOURCE"]).resolve()))
        from idol.sampler import sample_flow

        cls.mx, cls.np, cls.torch = mx, np, torch
        cls.mlx_sample, cls.time_grid = staticmethod(sample_euler), staticmethod(time_grid)
        cls.seeded_noise = staticmethod(seeded_noise)
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

    def test_guidance_modes_match_pytorch_and_each_other(self):
        mx, np, torch = self.mx, self.np, self.torch
        noise = np.array([[2., -1.], [1., 3.]], dtype=np.float32)
        cond = np.array([[0.5, -0.3], [0.7, 0.2]], dtype=np.float32)
        neg = np.array([[-0.2, 0.4], [-0.1, 0.9]], dtype=np.float32)
        for strength in (-1.0, 0.0, 0.5, 3.0):
            outputs = []
            for batched in (False, True):
                with self.subTest(strength=strength, batched=batched):
                    calls_t, calls_m = [], []
                    def torch_model(sample, time, condition):
                        calls_t.append(sample.shape[0])
                        return 0.1 * sample + time[:, None] * 0.001 + condition
                    def mlx_model(sample, time, condition):
                        calls_m.append(sample.shape[0])
                        return 0.1 * sample + time[:, None] * 0.001 + condition
                    expected = self.torch_sample(torch_model, torch.from_numpy(noise), torch.from_numpy(cond),
                                                 torch.from_numpy(neg), steps=4, rescale_t=3.0,
                                                 cfg_strength=strength, batch_cfg=batched, verbose=False).numpy()
                    actual = self.mlx_sample(mlx_model, mx.array(noise), mx.array(cond), steps=4,
                                             neg_condition=mx.array(neg), cfg_strength=strength, batch_cfg=batched)
                    mx.eval(actual)
                    self.assertEqual(calls_m, calls_t)
                    self.assertEqual(len(calls_m), 8 if strength > 0 and not batched else 4)
                    self.assertEqual(calls_m[0], 4 if strength > 0 and batched else 2)
                    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-4, atol=1e-5)
                    outputs.append(np.array(actual))
            np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-4, atol=1e-5)

    def test_guidance_input_validation(self):
        mx = self.mx
        sample = mx.array([[1.]])
        model = lambda sample, time, condition: condition
        for strength in (float("nan"), float("inf"), True, "bad"):
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                self.mlx_sample(model, sample, sample, cfg_strength=strength)
        with self.assertRaisesRegex(ValueError, "negative condition"):
            self.mlx_sample(model, sample, sample, cfg_strength=1.0)
        with self.assertRaisesRegex(ValueError, "batch_cfg"):
            self.mlx_sample(model, sample, sample, batch_cfg=1)

    def test_seeded_noise_and_sampling_are_reproducible(self):
        mx, np = self.mx, self.np
        condition = mx.array([[0.2, -0.3], [0.1, 0.4]])
        noise_a = self.seeded_noise((2, 2), 42)
        noise_b = self.seeded_noise((2, 2), 42)
        noise_c = self.seeded_noise((2, 2), 43)
        model = lambda sample, time, condition: sample * 0.1 + condition
        output_a = self.mlx_sample(model, noise_a, condition, steps=4)
        output_b = self.mlx_sample(model, noise_b, condition, steps=4)
        mx.eval(noise_a, noise_b, noise_c, output_a, output_b)
        np.testing.assert_array_equal(np.array(noise_a), np.array(noise_b))
        self.assertFalse(np.array_equal(np.array(noise_a), np.array(noise_c)))
        np.testing.assert_array_equal(np.array(output_a), np.array(output_b))
        explicit_a = self.mlx_sample(model, noise_a, condition, steps=4)
        mx.eval(explicit_a)
        np.testing.assert_array_equal(np.array(output_a), np.array(explicit_a))
        for dtype in ("float32", "float16", "bfloat16"):
            with self.subTest(dtype=dtype):
                self.assertEqual(self.seeded_noise((2, 2), 0, dtype=dtype).dtype, getattr(mx, dtype))

    def test_seeded_noise_does_not_advance_global_random_stream(self):
        mx, np = self.mx, self.np
        try:
            mx.random.seed(333)
            expected_first, expected_second = mx.random.normal((4,)), mx.random.normal((4,))
            mx.eval(expected_first, expected_second)
            mx.random.seed(333)
            first = mx.random.normal((4,))
            local = self.seeded_noise((4,), 17)
            second = mx.random.normal((4,))
            mx.eval(first, local, second)
            np.testing.assert_array_equal(np.array(first), np.array(expected_first))
            np.testing.assert_array_equal(np.array(second), np.array(expected_second))
        finally:
            mx.random.seed(0)

    def test_seeded_noise_rejects_invalid_configuration(self):
        for shape, seed, dtype in (((0,), 0, "float32"), ((True,), 0, "float32"),
                                   ([2], 0, "float32"), ((), 0, "float32"),
                                   ((2,), -1, "float32"), ((2,), 2**64, "float32"),
                                   ((2,), True, "float32"), ((2,), 0, "float64")):
            with self.subTest(shape=shape, seed=seed, dtype=dtype), self.assertRaises(ValueError):
                self.seeded_noise(shape, seed, dtype=dtype)
