# Preliminary spatial embeddings

[Español](../es/spatial-embeddings.md)

The two point-coordinate embeddings in `idol_mlx.layers` reproduce the
[pinned upstream SDK](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/layers.py):

- `PcdAbsolutePositionEmbedder` embeds coordinates per token with the
  denoiser's power-of-two frequency schedule, including its extended-frequency
  rule; FP32/FP16 inputs produce FP32 output in the tested cases.
- `PcdAbsolutePositionEmbedderv2` embeds coordinates per sample using the
  decoder's evenly spaced log frequencies and input dtype.

Both concatenate sine/cosine features in upstream order and pad to the requested
channel count when required. Frequencies are derived from configuration at each
call; no learned parameters or checkpoint loading are introduced.

Synthetic PyTorch parity tests use shared coordinates. V1 covers batches of
multiple tokens, padding, extended frequencies and FP32/FP16 inputs, with
`atol=1e-5, rtol=1e-4`. V2 covers multiple samples, padding and FP32/FP16
inputs; FP32 uses those same tolerances and FP16 uses `atol=3e-3, rtol=3e-3`.
These provisional tolerances and cases do not validate BF16, all shapes,
trained models or the full pipeline.

With MLX, PyTorch and the pinned upstream source available:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_layers_mlx.py' -v
```

Set `IDOL_TEST_MLX_DEVICE=gpu` to run MLX on Metal when available; PyTorch
runs on CPU. CI runs this test file on MLX CPU. See also the
[scalar sinusoidal embedding](sinusoidal-embedding.md).
