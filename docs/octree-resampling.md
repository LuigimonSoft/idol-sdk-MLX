# Systematic octree resampling primitive

[Español](../es/octree-resampling.md)

`idol_mlx.octree_sampling.systematic_resample` allocates a count per parent to
its children using systematic resampling. Inputs are MLX arrays: probabilities
`(..., bins)`, integer counts `(...)`, and explicit FP32 offsets `(...)` in
`[0, 1)`. The result is an int64 array of shape `(..., bins)` whose final-axis
sum equals the input count for each row. Octree cells use eight bins, but the
primitive supports any positive bin count.

```python
import mlx.core as mx
from idol_mlx.octree_sampling import systematic_resample

probabilities = mx.array([[0.1, 0.2, 0.3, 0.4]])
counts = mx.array([10])
offsets = mx.random.uniform(shape=(1,), key=mx.random.key(42))
allocation = systematic_resample(probabilities, counts, offsets=offsets)
```

The caller supplies independent offsets and controls the random key. An MLX
seed does not reproduce PyTorch's random stream; compare the implementations
using the same offsets. The function clamps negative probabilities to zero,
divides by `max(row_sum, 1)` and uses a uniform fallback for zero-sum rows,
matching the [pinned reference](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/decoder.py).
For row sums below one the remaining mass is assigned to the final bin, as in
the reference; callers wanting a normalized distribution must supply one.

Input validation synchronizes MLX evaluation and rejects non-finite values,
non-integer counts, invalid shapes, offsets outside `[0, 1)`, and counts above
`2**24 - 1`; the limit bounds the FP32 sample grid and fixed search iterations.
The implementation avoids allocating a per-point grid, using a fixed 24-step
binary search per bin instead. It has not been benchmarked on full-size octrees.

Tests compare controlled offsets against the upstream PyTorch implementation,
including boundary offsets, zero/negative/unnormalized rows, varied counts and
batches. A 4096-row test checks independent offsets, exact reference counts and
empirical frequencies. CI executes these tests with MLX CPU; they have also run
locally with Metal. This is only the allocation primitive: variable-size tree
traversal, leaf jitter, full anchor counts and end-to-end inference remain work
for issue #32.
