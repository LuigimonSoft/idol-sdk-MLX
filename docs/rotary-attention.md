# Preliminary denoiser rotary attention

[Español](../es/rotary-attention.md)

`idol_mlx.layers.RopeMultiHeadAttention` ports the denoiser's self-attention
from the [pinned upstream SDK](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/layers.py).
It projects queries, keys and values, rotates queries and keys using real
cosine/sine pairs, applies per-head normalization, computes scaled dot-product
attention and projects the result. The scale is `1/sqrt(head_dim)`. As with
the upstream block, this interface does not accept an attention mask.

The rotary argument uses the real-valued representation described in
[rotary and RePo](rotary-repo.md); the upstream complex representation is not
implemented. Weights are not loaded or converted automatically.

Synthetic tests use identical PyTorch/MLX weights and nontrivial query/key
normalization scales. They compare FP32/FP16 inputs, one and two heads,
nontrivial per-token rotations and identity rotations. They also confirm that
the nontrivial rotations affect the PyTorch output. Provisional tolerances are
`atol=1e-5, rtol=1e-4` for FP32 and `atol=3e-3, rtol=3e-3` for FP16. These
cases do not validate BF16, masked attention, trained checkpoints, every shape,
RePo-to-attention integration or full-model inference.

With MLX, PyTorch and the pinned upstream source installed:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_layers_mlx.py' -v
```

Set `IDOL_TEST_MLX_DEVICE=gpu` for MLX Metal when available; PyTorch stays on
CPU. CI runs the same shared-layer parity tests on MLX CPU.
