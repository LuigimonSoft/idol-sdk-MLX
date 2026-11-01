"""Controlled systematic-resampling checks against the pinned upstream SDK."""

import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("mlx") and os.environ.get("IDOL_REFERENCE_SOURCE"),
                     "MLX and pinned reference source are required")
class SystematicResamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        import numpy as np
        import torch
        from idol_mlx.octree_sampling import systematic_resample

        sys.path.insert(0, str(Path(os.environ["IDOL_REFERENCE_SOURCE"]).resolve()))
        from idol.models.decoder import systematic_sample

        cls.mx, cls.np, cls.torch = mx, np, torch
        cls.sample, cls.reference = staticmethod(systematic_resample), staticmethod(systematic_sample)
        device = os.environ.get("IDOL_TEST_MLX_DEVICE", "cpu")
        if device not in ("cpu", "gpu"):
            raise ValueError("IDOL_TEST_MLX_DEVICE must be cpu or gpu")
        previous = mx.default_device()
        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
        cls.addClassCleanup(mx.set_default_device, previous)

    def compare_reference(self, probabilities, counts, offset):
        mx, np, torch = self.mx, self.np, self.torch
        probabilities = np.asarray(probabilities, dtype=np.float32)
        counts = np.asarray(counts, dtype=np.int64)
        def fixed_rand(*size, device=None):
            return torch.full(size, offset, dtype=torch.float32, device=device)
        with patch("torch.rand", side_effect=fixed_rand):
            expected = self.reference(torch.from_numpy(probabilities.copy()), torch.from_numpy(counts.copy()))
        actual = self.sample(mx.array(probabilities), mx.array(counts),
                             offsets=mx.full(counts.shape, offset, dtype=mx.float32))
        mx.eval(actual)
        values = np.array(actual)
        np.testing.assert_array_equal(values, expected.numpy())
        np.testing.assert_array_equal(values.sum(axis=-1), counts)
        self.assertTrue((values >= 0).all())

    def test_zero_one_hot_unnormalized_and_boundary_offsets(self):
        probs = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
                 [1, 2, 3, 4, 5, 6, 7, 8], [-1, 0.1, 0.2, 0, 0, 0, 0, 0],
                 [0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0]]
        for offset in (0.0, 0.125, 0.5, 0.99999994):
            with self.subTest(offset=offset):
                self.compare_reference(probs, [8, 7, 29, 17, 4], offset)

    def test_batches_and_varied_counts(self):
        np = self.np
        rng = np.random.default_rng(11)
        probs = rng.random((2, 12, 8), dtype=np.float32)
        counts = rng.integers(0, 300, size=(2, 12), dtype=np.int64)
        for offset in (0.0, 0.123, 0.5, 0.99999994):
            with self.subTest(offset=offset):
                self.compare_reference(probs, counts, offset)

    def test_independent_offsets_and_empirical_distribution(self):
        mx, np, torch = self.mx, self.np, self.torch
        rng = np.random.default_rng(25)
        rows, count = 4096, 11
        probabilities = np.array([0.03, 0.07, 0.1, 0.2, 0.25, 0.15, 0.1, 0.1], dtype=np.float32)
        tiled = np.broadcast_to(probabilities, (rows, 8)).copy()
        offsets = rng.random(rows, dtype=np.float32)
        counts = np.full((rows,), count, dtype=np.int64)

        def fixed_rand(*size, device=None):
            self.assertEqual(size, (rows, 1))
            return torch.from_numpy(offsets.reshape(rows, 1)).to(device=device)

        with patch("torch.rand", side_effect=fixed_rand):
            expected = self.reference(torch.from_numpy(tiled.copy()), torch.from_numpy(counts))
        result = self.sample(mx.array(tiled), mx.array(counts), offsets=mx.array(offsets))
        mx.eval(result)
        actual = np.array(result)
        np.testing.assert_array_equal(actual, expected.numpy())
        np.testing.assert_array_equal(actual.sum(axis=-1), counts)
        np.testing.assert_allclose(actual.sum(axis=0) / (rows * count), probabilities, atol=0.004, rtol=0)

    def test_large_count_conservation(self):
        mx, np = self.mx, self.np
        count = 2**24 - 1
        probs = mx.array([[0.1, 0.2, 0.3, 0.4]])
        result = self.sample(probs, mx.array([count]), offsets=mx.array([0.5]))
        mx.eval(result)
        values = np.array(result)
        self.assertTrue((values >= 0).all())
        self.assertEqual(int(values.sum()), count)

    def test_all_supported_integer_count_types(self):
        mx, np = self.mx, self.np
        probs = mx.array([[0.5, 0.5], [0.5, 0.5]])
        offsets = mx.array([0.5, 0.5])
        for dtype in (mx.int8, mx.int16, mx.int32, mx.int64,
                      mx.uint8, mx.uint16, mx.uint32, mx.uint64):
            with self.subTest(dtype=dtype):
                result = self.sample(probs, mx.array([0, 7], dtype=dtype), offsets=offsets)
                mx.eval(result)
                np.testing.assert_array_equal(np.array(result), [[0, 0], [4, 3]])
        for dtype in (mx.int8, mx.int16, mx.int32, mx.int64):
            with self.subTest(negative_dtype=dtype), self.assertRaises(ValueError):
                self.sample(probs, mx.array([0, -1], dtype=dtype), offsets=offsets)
        for dtype in (mx.int32, mx.int64, mx.uint32, mx.uint64):
            with self.subTest(large_dtype=dtype), self.assertRaises(ValueError):
                self.sample(probs, mx.array([0, 2**24], dtype=dtype), offsets=offsets)
        with self.assertRaises(ValueError):
            self.sample(probs, mx.array(np.array([0, 2**64 - 1], dtype=np.uint64)), offsets=offsets)

    def test_rejects_invalid_inputs(self):
        mx = self.mx
        sample = self.sample
        probs, counts, offsets = mx.ones((2, 8)), mx.array([1, 2]), mx.array([0.1, 0.2])
        invalid = [(mx.ones((2, 7)), mx.array([1]), offsets), (probs, counts, mx.ones((1,))),
                   (probs, mx.array([1.0, 2.0]), offsets), (probs, mx.array([-1, 2]), offsets),
                   (probs, mx.array([1, 2**24]), offsets), (probs, counts, mx.array([0.1, 1.])),
                   (probs, counts, mx.array([0.1, float("nan")])),
                   (mx.full((2, 8), float("nan")), counts, offsets)]
        for args in invalid:
            with self.subTest(shapes=[a.shape for a in args]), self.assertRaises(ValueError):
                sample(args[0], args[1], offsets=args[2])
