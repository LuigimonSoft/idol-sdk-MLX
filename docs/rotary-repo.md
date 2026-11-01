# Preliminary rotary and RePo 3D operations

[Español](../es/rotary-repo.md)

The `idol_mlx.layers` module ports the real-valued path of
[`apply_rotary_emb` and `RePo3DRotaryEmbedding`](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/layers.py)
from the pinned upstream SDK. RePo predicts per-head 3D offsets using LayerNorm,
SiLU and three bias-free linear layers; learned frequency vectors convert these
into angles and `(cos, sin)` pairs. Rotary application treats adjacent feature
values as real/imaginary pairs and restores the original input dtype. Frequencies
can broadcast across heads, as in the upstream real-valued path.

The MLX port returns real cosine/sine pairs only; the upstream complex-valued
representation is not implemented. Frequency vectors and linear weights are
parameters and must be loaded with explicit schema and mapping; no trained
checkpoint conversion is included. The forward expression for `clamp_mul`
preserves the original stop-gradient structure, but training gradients are not
validated by this work.

Synthetic tests compare against the pinned PyTorch reference using identical
weights and frequencies for two head dimensions, and test rotary application
with broadcast and per-head frequencies and FP32/FP16 input. The provisional
tolerances are `atol=1e-5` and `rtol=1e-4`. These cases do not validate all
shapes, BF16, trained weights, attention integration or full-model inference.

With MLX, PyTorch and the pinned upstream source installed:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_layers_mlx.py' -v
```

Set `IDOL_TEST_MLX_DEVICE=gpu` to test MLX on Metal where available; PyTorch
runs on CPU. CI runs the shared-layer parity tests on MLX CPU.
