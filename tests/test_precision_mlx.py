"""Optional runtime checks; run with the MLX environment on Apple Silicon."""

import importlib.util
import unittest

from idol_mlx.precision import PrecisionPolicy, evaluate_boundary


@unittest.skipUnless(importlib.util.find_spec("mlx"), "MLX is not installed")
class PrecisionMLXTests(unittest.TestCase):
    def test_float_casts_and_boundary_on_mlx_arrays(self):
        import mlx.core as mx

        source = mx.array([1.0, -2.0], dtype=mx.float32)
        for name in ("float32", "float16", "bfloat16"):
            with self.subTest(dtype=name):
                result = PrecisionPolicy(name).cast(source)
                self.assertEqual(result.dtype, getattr(mx, name))
                evaluate_boundary("output", result)
                self.assertEqual(result.astype(mx.float32).tolist(), [1.0, -2.0])
        with self.assertRaises(TypeError):
            PrecisionPolicy().cast(mx.array([1], dtype=mx.int32))
