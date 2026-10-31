import unittest

import numpy as np

from idol_mlx.conversion import TensorMapping, convert_tensors
from idol_mlx.weights import CheckpointError, TensorSpec


class ConversionTests(unittest.TestCase):
    def test_identity_mapping_preserves_values_and_dtype(self):
        value = np.arange(6, dtype=np.float32).reshape(2, 3)
        result = convert_tensors({"upstream.weight": value},
                                 [TensorMapping("upstream.weight", "layer.weight")],
                                 {"layer.weight": TensorSpec((2, 3), "float32")})
        np.testing.assert_array_equal(result["layer.weight"], value)
        self.assertEqual(result["layer.weight"].dtype, np.float32)

    def test_conv2d_mapping_preserves_element_coordinates(self):
        value = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
        result = convert_tensors({"conv.weight": value},
                                 [TensorMapping("conv.weight", "weight", "conv2d_oihw_to_ohwi")],
                                 {"weight": TensorSpec((2, 4, 5, 3), "float32")})["weight"]
        for o, i, h, w in np.ndindex(value.shape):
            self.assertEqual(result[o, h, w, i], value[o, i, h, w])

    def test_duplicate_and_incomplete_mappings_fail(self):
        value = np.zeros((2,), dtype=np.float32)
        schema = {"w": TensorSpec((2,), "float32")}
        for rules in ([TensorMapping("x", "w"), TensorMapping("x", "v")],
                      [TensorMapping("x", "w"), TensorMapping("y", "w")]):
            with self.subTest(rules=rules), self.assertRaises(ValueError):
                convert_tensors({"x": value}, rules, schema)
        with self.assertRaisesRegex(CheckpointError, "unexpected"):
            convert_tensors({"x": value, "extra": value}, [TensorMapping("x", "w")], schema)
        with self.assertRaisesRegex(CheckpointError, "missing"):
            convert_tensors({"other": value}, [TensorMapping("x", "w")], schema)

    def test_invalid_layout_and_rank_fail(self):
        with self.assertRaises(ValueError):
            TensorMapping("x", "w", "unknown")
        with self.assertRaisesRegex(CheckpointError, "rank-4"):
            convert_tensors({"x": np.zeros(2)}, [TensorMapping("x", "w", "conv2d_oihw_to_ohwi")],
                            {"w": TensorSpec((2,), "float64")})

    def test_converted_output_requires_matching_schema(self):
        value = np.zeros((2, 3), dtype=np.float32)
        with self.assertRaisesRegex(CheckpointError, "Shape mismatch"):
            convert_tensors({"x": value}, [TensorMapping("x", "w")], {"w": TensorSpec((3, 2), "float32")})
        with self.assertRaisesRegex(CheckpointError, "Dtype mismatch"):
            convert_tensors({"x": value}, [TensorMapping("x", "w")], {"w": TensorSpec((2, 3), "float16")})


if __name__ == "__main__":
    unittest.main()
