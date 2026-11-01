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


def sample_octree(model, cond, *, num_points, level, seed):
    """Sample a batch of octree anchors using a local MLX random key.

    ``model(centers, resolutions, cond)`` returns eight logits per cell. The
    result has shape ``(batch, num_points, 3)`` in normalized coordinates.
    Variable-size cell packing requires eager device synchronization per level.
    """
    if not callable(model):
        raise TypeError("model must be callable.")
    if not isinstance(cond, mx.array) or cond.ndim != 3 or any(size < 1 for size in cond.shape):
        raise ValueError("cond must be a non-empty MLX array of shape (batch, tokens, channels).")
    if cond.dtype != mx.float32:
        raise ValueError("cond must have float32 dtype.")
    if isinstance(num_points, bool) or not isinstance(num_points, int) or not 0 <= num_points < 2**24:
        raise ValueError("num_points must be an integer in [0, 2**24 - 1].")
    if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= 23:
        raise ValueError("level must be an integer in [0, 23].")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer.")
    batch = cond.shape[0]
    if num_points == 0:
        return mx.zeros((batch, 0, 3), dtype=mx.float32)
    coords = mx.zeros((batch, 1, 3), dtype=mx.int32)
    counts = mx.full((batch, 1), num_points, dtype=mx.int64)
    child_offsets = mx.array([[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)], dtype=mx.int32)
    key = mx.random.key(seed)
    for current in range(1, level + 1):
        centers = (coords.astype(mx.float32) + 0.5) / (1 << (current - 1))
        resolutions = mx.full((batch,), 1 << current, dtype=mx.int32)
        logits = model(centers, resolutions, cond)
        if not isinstance(logits, mx.array) or logits.shape != (*counts.shape, 8):
            raise ValueError("model must return an MLX array of shape (batch, cells, 8).")
        key, allocation_key = mx.random.split(key)
        offsets = mx.random.uniform(shape=counts.shape, dtype=mx.float32, key=allocation_key)
        children = systematic_resample(mx.softmax(logits, axis=-1), counts, offsets=offsets)
        children = children.reshape(batch, -1)
        child_coords = (coords[:, :, None, :] * 2 + child_offsets[None, None, :, :]).reshape(batch, -1, 3)
        mask = children > 0
        max_valid = int(mx.max(mx.sum(mask, axis=1)).item())
        # Stable sorting packs non-empty children first, preserving source order.
        order = mx.argsort((~mask).astype(mx.int32), axis=1)[:, :max_valid]
        counts = mx.take_along_axis(children, order, axis=1)
        coords = mx.take_along_axis(child_coords, order[..., None], axis=1)
        coords = mx.where(counts[..., None] > 0, coords, 0)
    # Expand each batch independently, as repeat with array counts is unavailable.
    indices = mx.arange(num_points, dtype=mx.int64)
    anchors = []
    for index in range(batch):
        positions = mx.searchsorted(mx.cumsum(counts[index]), indices, side="right")
        anchors.append(coords[index][positions])
    cells = mx.stack(anchors, axis=0)
    key, jitter_key = mx.random.split(key)
    jitter = mx.random.uniform(shape=cells.shape, dtype=mx.float32, key=jitter_key)
    return (cells.astype(mx.float32) + jitter) / (1 << level)
