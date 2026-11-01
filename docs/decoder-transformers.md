# Preliminary decoder transformer blocks

[Español](../es/decoder-transformers.md)

The following `idol_mlx.layers` blocks port the
[pinned upstream SDK](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/layers.py):

- `TransformerCrossBlock` combines self-attention, cross-attention and a
  feed-forward network with normalization and residual connections.
- `ModulatedTransformerCrossOnlyBlock` uses cross-attention and a feed-forward
  network, modulated by six shift/scale/gate components for octree decoding.

Synthetic FP32 parity tests compare each block with PyTorch using identical
inputs, context and all parameters. The test maps sequential parameter names
and checks shapes before loading; this is test code, not an automatic trained
checkpoint converter. Tests use different query/context lengths and a changed
context, and verify zero modulation gates preserve the input for the modulated
block. Provisional tolerances are `atol=1e-5, rtol=1e-4`.

FP16/BF16, trained weights, all shapes and full-model integration are not
validated. These blocks expose no attention masks, matching the upstream API.
See also [decoder attention](decoder-attention.md) and the
[unified transformer](unified-transformer.md).

With MLX, PyTorch and the pinned source installed:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_layers_mlx.py' -v
```

Set `IDOL_TEST_MLX_DEVICE=gpu` for MLX Metal when available; PyTorch stays on
CPU. CI runs shared-layer parity tests on MLX CPU.
