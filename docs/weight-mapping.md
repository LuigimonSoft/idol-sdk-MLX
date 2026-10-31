# Preliminary tensor mapping

[Español](../es/weight-mapping.md)

`idol_mlx.conversion` provides an in-memory, explicit, one-to-one mapping of
already loaded tensors. Each rule names the source and target and selects either
`identity` or `conv2d_oihw_to_ohwi`. The latter transposes a standard PyTorch
Conv2d kernel from output-input-height-width to MLX's
output-height-width-input layout. The caller must identify compatible layers;
grouped, transposed and other convolution variants have not been validated.
See the [PyTorch Conv2d](https://docs.pytorch.org/docs/main/generated/torch.nn.Conv2d.html)
and [MLX conv2d](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.conv2d.html) documentation.

```python
from idol_mlx.conversion import TensorMapping, convert_tensors
from idol_mlx.weights import TensorSpec

# tensors contains a previously loaded MLX array under "conv.weight".
converted = convert_tensors(
    tensors,
    [TensorMapping("conv.weight", "weight", "conv2d_oihw_to_ohwi")],
    {"weight": TensorSpec((2, 3, 3, 4), "float32")},
)
```

This example expects a source kernel shaped `(2, 4, 3, 3)`. Every input and
output name must appear exactly once, and each result must match the supplied
shape and dtype. Conversion does not cast tensors or infer schemas, names or
layer semantics. The caller supplies trustworthy inputs and a schema derived
from the actual target model.

A synthetic, ungrouped float32 convolution was compared with PyTorch on CPU
and MLX on Metal; its maximum absolute output difference was approximately
`1.91e-6`. This single example is not a model-level parity guarantee.
Model-specific mappings, trained-weight validation, file export and an
end-to-end converter remain future work.
