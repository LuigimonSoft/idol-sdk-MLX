import unittest
from types import SimpleNamespace
from unittest.mock import patch

from idol_mlx.weights import CheckpointError, TensorSpec, load_weights


def tensor(shape, dtype):
    return SimpleNamespace(shape=shape, dtype="mlx.core." + dtype)


class WeightTests(unittest.TestCase):
    def test_matching_complete_schema_returns_tensors(self):
        tensors = {"linear.weight": tensor((2, 3), "float16"), "offset": tensor((), "float32")}
        expected = {"linear.weight": TensorSpec((2, 3), "float16"), "offset": TensorSpec((), "float32")}
        with patch("idol_mlx.weights._read_safetensors", return_value=tensors) as read:
            self.assertIs(load_weights("weights.safetensors", expected), tensors)
        read.assert_called_once()

    def test_missing_or_unexpected_keys_fail(self):
        with patch("idol_mlx.weights._read_safetensors", return_value={"extra": tensor((2,), "float32")}):
            with self.assertRaisesRegex(CheckpointError, "missing=.*weight.*unexpected=.*extra"):
                load_weights("weights.safetensors", {"weight": TensorSpec((2,), "float32")})

    def test_shape_and_dtype_mismatch_fail(self):
        for actual, pattern in [(tensor((3,), "float32"), "Shape mismatch"),
                                (tensor((2,), "float16"), "Dtype mismatch")]:
            with self.subTest(pattern=pattern), patch("idol_mlx.weights._read_safetensors", return_value={"w": actual}):
                with self.assertRaisesRegex(CheckpointError, pattern):
                    load_weights("weights.safetensors", {"w": TensorSpec((2,), "float32")})

    def test_file_type_and_empty_schema_rejected(self):
        with self.assertRaises(CheckpointError):
            load_weights("weights.npz", {"w": TensorSpec((2,), "float32")})
        with self.assertRaises(ValueError):
            load_weights("weights.safetensors", {})

    def test_invalid_tensor_specs_rejected(self):
        for shape, dtype in [((-1,), "float32"), ((True,), "float32"), ([1], "float32"), ((1,), "")]:
            with self.subTest(shape=shape, dtype=dtype), self.assertRaises(ValueError):
                TensorSpec(shape, dtype)

    def test_read_failures_are_reported(self):
        with patch("idol_mlx.weights._read_safetensors", side_effect=OSError("bad file")):
            with self.assertRaisesRegex(CheckpointError, "Cannot read"):
                load_weights("weights.safetensors", {"w": TensorSpec((2,), "float32")})


if __name__ == "__main__":
    unittest.main()
