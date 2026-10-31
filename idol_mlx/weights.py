"""Load local MLX safetensors with explicit shape and dtype expectations."""

from dataclasses import dataclass
from pathlib import Path


class CheckpointError(ValueError):
    """The local checkpoint cannot be read or does not match its schema."""


@dataclass(frozen=True)
class TensorSpec:
    """Expected shape and canonical MLX dtype name, for example 'float32'."""

    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self):
        if not isinstance(self.shape, tuple) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in self.shape
        ):
            raise ValueError("TensorSpec.shape must be a tuple of non-negative integers.")
        if not isinstance(self.dtype, str) or not self.dtype or not self.dtype.isidentifier():
            raise ValueError("TensorSpec.dtype must be a canonical dtype name.")


def _validate_schema(expected):
    if not isinstance(expected, dict) or not expected or any(
        not isinstance(key, str) or not key or not isinstance(spec, TensorSpec)
        for key, spec in expected.items()
    ):
        raise ValueError("Expected schema must be a non-empty dictionary of names to TensorSpec.")


def _read_safetensors(path):
    # Import only when needed, so config and schema validation need no GPU.
    import mlx.core as mx

    return mx.load(str(path))


def load_weights(path, expected):
    """Read a trusted local safetensors file and enforce its complete schema.

    The caller supplies every expected key and TensorSpec. No name mapping,
    layout conversion or dtype casting is performed.
    """
    path = Path(path)
    if path.suffix != ".safetensors":
        raise CheckpointError("Checkpoint must be a .safetensors file.")
    _validate_schema(expected)
    try:
        tensors = _read_safetensors(path)
    except (OSError, ValueError, RuntimeError) as error:
        raise CheckpointError("Cannot read the safetensors checkpoint.") from error
    return validate_tensors(tensors, expected)


def validate_tensors(tensors, expected):
    """Validate an in-memory tensor dictionary against a complete schema."""
    _validate_schema(expected)
    if not isinstance(tensors, dict):
        raise CheckpointError("The checkpoint did not contain a tensor dictionary.")
    missing = sorted(expected.keys() - tensors.keys())
    unexpected = sorted(tensors.keys() - expected.keys())
    if missing or unexpected:
        raise CheckpointError(f"Checkpoint keys do not match: missing={missing}, unexpected={unexpected}.")
    for name, spec in expected.items():
        tensor = tensors[name]
        if tuple(tensor.shape) != spec.shape:
            raise CheckpointError(f"Shape mismatch for {name}: expected {spec.shape}, got {tuple(tensor.shape)}.")
        dtype = str(tensor.dtype).removeprefix("mlx.core.")
        if dtype != spec.dtype:
            raise CheckpointError(f"Dtype mismatch for {name}: expected {spec.dtype}, got {dtype}.")
    return tensors
