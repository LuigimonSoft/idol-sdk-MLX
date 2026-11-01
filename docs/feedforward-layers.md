# Preliminary MLX feed-forward layers

[Español](../es/feedforward-layers.md)

`idol_mlx.layers.FeedForwardNet` and `MLP` reproduce the linear/GELU block
structure in the [pinned upstream SDK](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/layers.py).
They use biased MLX linear layers and `mlx.nn.gelu_approx`, corresponding to
PyTorch `GELU(approximate="tanh")`. Linear weights have shape
`(output_features, input_features)`. This does not automatically map a checkpoint:
upstream sequential names such as `mlp.0.weight` differ from MLX names such as
`mlp.layers.0.weight`; use explicit rules and validate each target schema.

The one-layer upstream `MLP` uses `inner_channels` as its final linear input
dimension even when it differs from `channels`. This port preserves that
behavior; callers must supply an input with the matching final dimension.

Synthetic parity tests copy the same initialized weights and biases from
PyTorch and use the same inputs for FP32 and FP16. They cover two expansion
ratios and MLP depths one, two and three; the input has multiple batches and
tokens. The exploratory tolerances are `atol=1e-5, rtol=1e-4` for FP32 and
`atol=2e-3, rtol=2e-3` for FP16. They do not validate trained checkpoints,
BF16, all input ranges or end-to-end quality. SiLU-based upstream blocks are
not covered by this change.

Use the combined reference and MLX environment and set the pinned reference
source path:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_layers_mlx.py' -v
```

Set `IDOL_TEST_MLX_DEVICE=gpu` to run MLX on Metal where available; PyTorch
remains on CPU. CI runs these tests on MLX CPU after installing the pinned MLX
version. See [normalization layers](normalization-layers.md) for the other
implemented shared blocks.
