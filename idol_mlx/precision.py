"""Explicit precision choices and evaluation boundaries for future MLX components."""

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class PrecisionPolicy:
    """Choose a floating-point storage dtype; reduced precision is experimental.

    This policy is opt-in at each call site; it neither changes MLX global state
    nor determines the internal accumulation precision of individual operations.
    """

    dtype: str = "float32"

    def __post_init__(self):
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("dtype must be float32, float16, or bfloat16.")

    def cast(self, tensor):
        """Cast an MLX floating-point array without forcing evaluation."""
        mx = import_module("mlx.core")
        if not isinstance(tensor, mx.array) or not mx.issubdtype(tensor.dtype, mx.floating):
            raise TypeError("Expected an MLX floating-point array.")
        return tensor.astype(getattr(mx, self.dtype))


def evaluate_boundary(boundary, *arrays):
    """Materialize arrays at an explicit component, sampling-step or output boundary."""
    if boundary not in ("component", "sampling_step", "output"):
        raise ValueError("Unknown evaluation boundary.")
    if not arrays:
        raise ValueError("At least one array is required.")
    mx = import_module("mlx.core")
    if any(not isinstance(array, mx.array) for array in arrays):
        raise TypeError("Evaluation boundaries accept MLX arrays only.")
    mx.eval(*arrays)
