# Preliminary decoder attention

[Español](../es/decoder-attention.md)

`idol_mlx.layers.MultiHeadAttention` ports the decoder's self- and
cross-attention from the [pinned upstream SDK](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/layers.py).
The query and key projections use the upstream per-head normalization, and
scaled dot-product attention uses `1/sqrt(head_dim)` with an output projection.
Cross-attention accepts a separate context with its own sequence length and
feature dimension. This layer, like the upstream layer, exposes no mask input;
masked attention is not validated here. Checkpoint loading is not automatic.

Synthetic parity tests copy the same weights, biases and nontrivial query/key
scales from PyTorch after checking parameter shapes. They use batch size 2,
query length 3, context length 5, two heads and FP32/FP16; cross-attention is
also checked with a changed context. Provisional tolerances are `atol=1e-5,
rtol=1e-4` in FP32 and `atol=3e-3, rtol=3e-3` in FP16. These cases do not
validate all shapes, masks, BF16, trained weights, denoiser rotary attention or
full-model inference.

With MLX, PyTorch and the pinned upstream source installed:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_layers_mlx.py' -v
```

Use `IDOL_TEST_MLX_DEVICE=gpu` for MLX Metal when available; PyTorch stays on
CPU. CI runs this parity test file on MLX CPU.
