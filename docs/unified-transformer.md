# Preliminary unified transformer block

[Español](../es/unified-transformer.md)

`idol_mlx.layers.UnifiedTransformerBlock` ports the denoiser/refiner block from
the [pinned upstream SDK](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/layers.py).
It combines LayerNorm32, real rotary self-attention, a feed-forward block and
residual connections. With modulation enabled, a learned shift table is added to
the supplied modulation and split into six shift/scale/gate components. Without
modulation, it uses affine normalization and direct residual updates.

Synthetic FP32 tests copy every parameter from a PyTorch block after checking
names and shapes, explicitly mapping PyTorch sequential layer names to MLX
names. Both modulation modes are compared using identical inputs, rotary pairs
and weights. The modulated test checks nonzero modulation changes the result
and zero effective gates preserve the input. Tolerance is `atol=1e-5,
rtol=1e-4`. This test-only mapping is not a trained-checkpoint converter.

The block has not been validated with FP16/BF16, trained weights, generated
RePo output, all shapes or full-model inference. Other transformer block types
remain pending.

With MLX, PyTorch and the pinned source installed:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_layers_mlx.py' -v
```

Set `IDOL_TEST_MLX_DEVICE=gpu` for MLX Metal when available; PyTorch stays on
CPU. CI runs the shared-layer parity tests on MLX CPU.
