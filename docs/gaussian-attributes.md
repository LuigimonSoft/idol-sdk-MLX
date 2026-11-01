# Gaussian attribute conversion

[Español](../es/gaussian-attributes.md)

`idol_mlx.gaussian_attributes.decode_gaussian_attributes` converts FP32 anchor
coordinates and raw feature outputs from the preliminary Gaussian decoder into
batched MLX arrays using the [pinned upstream conventions](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/decoder.py).
Provide the decoder's representation configuration, `points_offset_perturbation`
and `base_offset_scale` buffers after loading its checkpoint.

```python
from idol_mlx.gaussian_attributes import decode_gaussian_attributes

# model is a loaded GaussianFixedlenDecoder, args are its configuration arguments.
features = model(anchors, condition)
attributes = decode_gaussian_attributes(
    anchors, features, representation_config=args["representation_config"],
    points_offset_perturbation=model.points_offset_perturbation,
    base_offset_scale=model.base_offset_scale,
)
```

For batch `B`, anchors `L` and `K` Gaussians per anchor, the result contains:

| Key | Shape | Convention |
| --- | --- | --- |
| `xyz` | `[B, L*K, 3]` | z-up positions, applying the reference offset and subtracting 0.5 |
| `features_dc` | `[B, L*K, 1, 3]` | degree-zero spherical harmonic coefficients, not RGB |
| `scaling` | `[B, L*K, 3]` | softplus scale with configured bias and minimum kernel |
| `rotation` | `[B, L*K, 4]` | normalized `(w, x, y, z)` quaternion with identity offset |
| `opacity` | `[B, L*K, 1]` | sigmoid with configured opacity bias |

The upstream API returns a list of per-batch Gaussian objects; this API returns
batched arrays with equivalent per-item values. A zero pre-normalization
quaternion remains zero as in the reference. Position offsets are not clamped;
finite sigmoid outputs may saturate to 0 or 1. Inputs must be nonempty FP32 MLX
arrays with compatible shapes; configuration biases and buffer shapes are
validated. This function does not export PLY or rotate coordinates to y-up.

Synthetic tests compare all five attributes against PyTorch on CPU and Metal,
including multiple batches, per-anchor counts, degenerate quaternions, SH color
and saturation. With `decoder_gs.safetensors` from
[`speridlabs/idol` revision `2731c2291fc65a797c5cb627a5e5526fc1490902`](https://huggingface.co/speridlabs/idol/tree/2731c2291fc65a797c5cb627a5e5526fc1490902),
32 synthetic anchors, 8 context tokens and seed 42, all attributes passed
`atol=2e-5`, `rtol=2e-4` on CPU and Metal. On Metal the maximum absolute error
was `5.96046e-8` for position, `0` for SH color, `7.45058e-8` for scale, `0`
for rotation and `1.19209e-7` for opacity. A second Metal case (64 anchors,
16 context tokens, seed 123) also passed all five comparisons. Measurements
used Apple M4, macOS 27.2, Python 3.13.14, MLX 0.32.3 and PyTorch 2.15.0.

**The diagnostic supplies identical PyTorch-generated raw features to both
converters** to isolate attribute conversion. It does not validate the MLX
Gaussian network's own features, which still have a documented Metal parity
failure in [the Gaussian decoder diagnostic](gaussian-decoder.md), nor does it
validate PLY export, mixed precision or end-to-end generation. No weights or
inputs were committed.

To reproduce with trusted local files and the pinned reference on `PYTHONPATH`:

```sh
PYTHONPATH=../idol-sdk-reference .venv-reference/bin/python -m scripts.validate_gaussian_attributes \
  --config /trusted/path/config.json --checkpoint /trusted/path/decoder_gs.safetensors \
  --anchors 32 --context-tokens 8 --seed 42 --device gpu
```

Use `--device cpu` for MLX CPU. The diagnostic allocates both full models,
prints aggregate metrics and exits 0 only when all five comparisons pass.
