# Octree anchor sampling

[Español](../es/octree-traversal.md)

`idol_mlx.octree_sampling.sample_octree` descends the octree using a model
that returns eight child logits per cell, allocates counts using systematic
resampling and adds uniform jitter inside each leaf. It returns an MLX FP32
array of shape `[batch, num_points, 3]` in normalized coordinates. The model
receives cell centers `[B, cells, 3]`, integer resolutions `[B]` and FP32
conditioning `[B, tokens, channels]`. The preliminary octree network can be
used as the model.

```python
from idol_mlx.octree_sampling import sample_octree

# model is a loaded OctreeProbabilityFixedlenDecoder; cond is an MLX FP32 array.
anchors = sample_octree(model, cond, num_points=32, level=2, seed=101)
```

`num_points` must be in `[0, 2**24 - 1]`, `level` in `[0, 23]` and `seed` an
unsigned 64-bit integer. Zero points returns an empty array. A local MLX key
controls resampling and jitter; the same seed, inputs, device and MLX version
reproduce the result, but MLX and PyTorch do not share a random stream.
Coordinates are in `[0, 1)` mathematically; FP32 rounding near the upper edge
may produce `1.0` at deep levels.

Nonempty children are packed per batch using stable sorting. The largest active
cell count is read back to the host each level; outputs use padded batches and
the leaf expansion loops over batches. This is an eager, preliminary path, not
a fully GPU-resident implementation. Level and point limits do not guarantee
sufficient memory; see [measured octree scenarios](octree-profiling.md) for
observations and limitations on one machine.

Controlled tests compare the pinned PyTorch sampler with identical offsets and
jitter for multiple batches and levels 0–3. They also check counts, uniform
root occupancy, seed reproducibility and input validation on CPU and Metal.
With the trained octree checkpoint from `speridlabs/idol` revision
`2731c2291fc65a797c5cb627a5e5526fc1490902`, an FP32 reference, one synthetic
conditioning input, 32 points, level 2 and both offsets and jitter fixed to
`0.375`, all 96 coordinates matched exactly on MLX CPU and Metal. This single
case does not validate arbitrary trained inputs, random streams, mixed precision
or end-to-end image-to-3D generation. Weights and inputs were not committed.

With the pinned reference checkout on `PYTHONPATH` and trusted local files:

```sh
PYTHONPATH=../idol-sdk-reference .venv-reference/bin/python -m scripts.validate_octree_traversal \
  --config /trusted/path/config.json --checkpoint /trusted/path/decoder.safetensors --device gpu
```

Use `--device cpu` for MLX CPU. The diagnostic fixes offsets and jitter only
within the process, outputs aggregate coordinate metrics, and exits 0 only when
all coordinates match exactly. It loads the full trained decoder weights.
