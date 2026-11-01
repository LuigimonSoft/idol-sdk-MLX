# Preliminary MLX normalization layers

[Español](../es/normalization-layers.md)

`idol_mlx.layers` ports two shared layers from the upstream SDK at
[commit 4702a7f](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/layers.py):

- `LayerNorm32`: converts input to float32 for last-dimension layer
  normalization, then restores the input dtype. Supports the single feature
  dimension and optional affine parameters used by Idol; the MLX implementation
  requires a positive epsilon.
- `MultiHeadRMSNorm`: computes the float32 L2 norm per head with a minimum
  denominator of `1e-12`, multiplies by learned `gamma` and `sqrt(dim)`, then
  restores the input dtype. This is the original formula, not a generic RMSNorm.

Synthetic tests compare against the original PyTorch classes using identical
inputs and parameters, including affine and non-affine layer normalization,
nontrivial gamma, zeros, near-zero inputs, FP32 and FP16. Exploratory thresholds
are `atol=1e-5, rtol=1e-4` for FP32 and `atol=1e-3, rtol=1e-3` for FP16;
these are not whole-model acceptance thresholds. Tests cover only the cited
cases and do not validate trained weights or BF16.

With MLX and the pinned reference environment installed, run from the project
root with the upstream path configured:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_layers_mlx.py' -v
```

Set `IDOL_TEST_MLX_DEVICE=gpu` to run MLX on Metal when available; PyTorch
reference operations remain on CPU. CI installs the pinned MLX version after
its standard suite and runs these tests on MLX CPU. Linear layers, activations,
other operations and full-model inference remain to be ported or validated.
