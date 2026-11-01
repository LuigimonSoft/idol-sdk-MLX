"""Systematic allocation primitives for octree sampling."""

import mlx.core as mx


def systematic_resample(probabilities, counts, *, offsets):
    """Allocate integer counts to bins using one explicit uniform offset per row.

    ``probabilities`` has shape ``(..., bins)``; ``counts`` and ``offsets`` have
    shape ``(...)``. Offsets are FP32 values in ``[0, 1)``. Returns int64 bin
    counts. Values are validated eagerly; invalid input raises ``ValueError``.
    """
    if (not isinstance(probabilities, mx.array) or not isinstance(counts, mx.array) or
            not isinstance(offsets, mx.array)):
        raise ValueError("probabilities, counts and offsets must be MLX arrays.")
    if probabilities.ndim < 1 or probabilities.shape[-1] < 1 or probabilities.shape[:-1] != counts.shape:
        raise ValueError("probabilities must have shape counts.shape + (bins,).")
    if offsets.shape != counts.shape:
        raise ValueError("offsets must have the same shape as counts.")
    if probabilities.dtype not in (mx.float16, mx.bfloat16, mx.float32) or offsets.dtype != mx.float32:
        raise ValueError("probabilities must be floating point and offsets must be float32.")
    if counts.dtype not in (mx.int8, mx.int16, mx.int32, mx.int64, mx.uint8, mx.uint16, mx.uint32, mx.uint64):
        raise ValueError("counts must have an integer dtype.")
    if counts.size == 0:
        raise ValueError("counts must not be empty.")
    # Widen before comparing: MLX scalar comparisons otherwise narrow the limit
    # to the input dtype. Out-of-range uint64 values wrap negative and fail too.
    wide_counts = counts.astype(mx.int64)
    valid = (mx.all(mx.isfinite(probabilities)) & mx.all(mx.isfinite(offsets)) &
             mx.all((offsets >= 0) & (offsets < 1)) &
             mx.all((wide_counts >= 0) & (wide_counts <= 2**24 - 1)))
    if not bool(valid.item()):
        raise ValueError("inputs must be finite, offsets in [0, 1), and counts in [0, 2**24 - 1].")
    bins = probabilities.shape[-1]
    values = mx.maximum(probabilities.astype(mx.float32), 0)
    sums = mx.sum(values, axis=-1, keepdims=True)
    # Preserve upstream behavior for subunit sums as well as its zero-row fallback.
    values = mx.where(sums == 0, 1.0 / bins, values / mx.maximum(sums, 1))
    cdf = mx.minimum(mx.cumsum(values, axis=-1), 1.0)
    n = wide_counts[..., None]
    n_float = mx.maximum(n, 1).astype(mx.float32)
    start = offsets[..., None] / n_float
    # Upper bound of the upstream float32 grid: the number of samples <= cdf.
    low = mx.zeros(cdf.shape, dtype=mx.int64)
    high = mx.broadcast_to(n, cdf.shape)
    for _ in range(24):
        mid = (low + high) // 2
        point = start + mid.astype(mx.float32) / n_float
        below = point <= cdf
        low = mx.where((mid < high) & below, mid + 1, low)
        high = mx.where((mid < high) & ~below, mid, high)
    # Upstream clips search results into the last bin, including remaining mass.
    boundary = mx.concatenate([mx.zeros_like(n), low[..., :-1], n], axis=-1)
    return boundary[..., 1:] - boundary[..., :-1]
