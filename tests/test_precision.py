import unittest
from unittest.mock import MagicMock, patch

from idol_mlx.precision import PrecisionPolicy, evaluate_boundary


class FakeArray:
    def __init__(self, dtype="float32"):
        self.dtype = dtype
        self.cast_to = None

    def astype(self, dtype):
        self.cast_to = dtype
        return self


class PrecisionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.mx = MagicMock()
        self.mx.array = FakeArray
        self.mx.float32 = "float32"
        self.mx.float16 = "float16"
        self.mx.bfloat16 = "bfloat16"
        self.mx.issubdtype.side_effect = lambda dtype, _: dtype in ("float32", "float16", "bfloat16")
        self.loader = patch("idol_mlx.precision.import_module", return_value=self.mx)
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def test_default_and_explicit_dtypes_cast_without_evaluation(self):
        for name in ("float32", "float16", "bfloat16"):
            tensor = FakeArray()
            policy = PrecisionPolicy() if name == "float32" else PrecisionPolicy(name)
            self.assertIs(policy.cast(tensor), tensor)
            self.assertEqual(tensor.cast_to, name)
        self.mx.eval.assert_not_called()

    def test_invalid_dtype_and_nonfloating_inputs_are_rejected(self):
        for dtype in ("float64", "Float32", None):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                PrecisionPolicy(dtype)
        for value in (FakeArray("int32"), object()):
            with self.subTest(value=value), self.assertRaises(TypeError):
                PrecisionPolicy().cast(value)

    def test_evaluation_occurs_only_for_valid_explicit_boundaries(self):
        for boundary in ("component", "sampling_step", "output"):
            first, second = FakeArray(), FakeArray()
            evaluate_boundary(boundary, first, second)
            self.mx.eval.assert_called_with(first, second)
        self.assertEqual(self.mx.eval.call_count, 3)
        with self.assertRaises(ValueError):
            evaluate_boundary("layer", FakeArray())
        with self.assertRaises(ValueError):
            evaluate_boundary("output")
        with self.assertRaises(TypeError):
            evaluate_boundary("output", object())
        self.assertEqual(self.mx.eval.call_count, 3)


if __name__ == "__main__":
    unittest.main()
