"""Small, backend-independent metrics for paired numeric reference tensors."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParityResult:
    passed: bool
    finite: bool
    count: int
    failed_count: int
    max_absolute_error: float | None
    mean_absolute_error: float | None


def compare(reference, candidate, *, atol, rtol):
    """Apply abs(candidate-reference) <= atol + rtol*abs(reference).

    Inputs must be non-empty real numeric arrays of identical shape. Non-finite
    inputs always fail, including when both contain NaN or the same infinity.
    """
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be finite and non-negative")
    ref = np.asarray(reference)
    actual = np.asarray(candidate)
    if ref.shape != actual.shape:
        raise ValueError("tensor shapes must match")
    if ref.size == 0:
        raise ValueError("tensors must not be empty")
    for value in (ref, actual):
        if not np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.complexfloating):
            raise TypeError("tensors must contain real numeric values")
    # Integers outside this range are not all exactly representable in FP64.
    for value in (ref, actual):
        if np.issubdtype(value.dtype, np.integer):
            if int(value.min()) < -(2**53) or int(value.max()) > 2**53:
                raise ValueError("integer values must be within the exact FP64 range [-2**53, 2**53]")
    ref = ref.astype(np.float64)
    actual = actual.astype(np.float64)
    finite = np.isfinite(ref) & np.isfinite(actual)
    if not finite.all():
        return ParityResult(False, False, ref.size, int((~finite).sum()), None, None)
    with np.errstate(over="ignore", invalid="ignore"):
        error = np.abs(actual - ref)
        allowed = atol + rtol * np.abs(ref)
    matches = np.isfinite(error) & (error <= allowed)
    failures = int((~matches).sum())
    max_error = float(error.max())
    if max_error == 0 or not np.isfinite(max_error):
        mean_error = max_error
    else:
        mean_error = float(np.mean(error / max_error) * max_error)
    return ParityResult(failures == 0, True, ref.size, failures, max_error, mean_error)
