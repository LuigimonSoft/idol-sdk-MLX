import unittest

import numpy as np

from scripts.parity_metrics import compare


class ParityMetricsTests(unittest.TestCase):
    def test_exact_match(self):
        result = compare([0.0, 1.0], [0.0, 1.0], atol=0, rtol=0)
        self.assertTrue(result.passed)
        self.assertEqual(result.failed_count, 0)
        self.assertEqual(result.max_absolute_error, 0)

    def test_absolute_tolerance_near_zero(self):
        self.assertTrue(compare([0.0], [0.00001], atol=0.00001, rtol=0).passed)
        self.assertFalse(compare([0.0], [0.00002], atol=0.00001, rtol=0).passed)

    def test_relative_tolerance_and_summary(self):
        result = compare([100.0, 200.0], [100.5, 203.0], atol=0, rtol=0.01)
        self.assertFalse(result.passed)
        self.assertEqual(result.failed_count, 1)
        self.assertEqual(result.max_absolute_error, 3.0)
        self.assertEqual(result.mean_absolute_error, 1.75)

    def test_nonfinite_always_fails(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                result = compare([value], [value], atol=1, rtol=1)
                self.assertFalse(result.passed)
                self.assertFalse(result.finite)
                self.assertIsNone(result.max_absolute_error)

    def test_shape_and_empty_rejected(self):
        with self.assertRaises(ValueError):
            compare([1], [[1]], atol=1, rtol=1)
        with self.assertRaises(ValueError):
            compare([], [], atol=1, rtol=1)

    def test_invalid_tolerances_and_types_rejected(self):
        for value in (-1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                compare([1], [1], atol=value, rtol=0)
        with self.assertRaises(TypeError):
            compare([1 + 2j], [1 + 2j], atol=1, rtol=1)
        with self.assertRaises(TypeError):
            compare(['x'], ['x'], atol=1, rtol=1)


if __name__ == '__main__':
    unittest.main()
