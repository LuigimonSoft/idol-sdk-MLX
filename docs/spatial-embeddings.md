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
channel count when required. For `freq_dim=170, max_res=16`, the schedule is a
canonical FP32 constant generated from the pinned upstream position embedder on
PyTorch 2.15.0 CPU; it contains no checkpoint data. This avoids backend-specific
rounding differences that are amplified by high-frequency trigonometry. Other
V1 configurations use the general MLX calculation. No learned parameters, PyTorch
inference dependency or checkpoint loading are introduced. The canonical values
can be reproduced with `PcdAbsolutePositionEmbedder(1024, in_channels=3,
max_res=16).freqs` in the pinned reference environment; the regression test
checks every value exactly. For V2 with `freq_dim=170, max_res=10`, the
canonical decoder frequencies are similarly generated from
`PcdAbsolutePositionEmbedderv2(1024, in_channels=3, max_res=10).freqs` and
cast to the input dtype as in the reference; other V2 configurations use the
general MLX calculation. This V2 embedding is also used by
the Gaussian decoder, whose full-network parity is not validated by this change.

Synthetic PyTorch parity tests use shared coordinates. V1 covers batches of
multiple tokens, padding, extended frequencies and FP32/FP16 inputs, with
`atol=1e-5, rtol=1e-4`. V2 covers multiple samples, padding, fractional interior frequency exponents
and FP32/FP16 inputs; FP32 uses those same tolerances and FP16 uses `atol=3e-3, rtol=3e-3`.
These provisional tolerances and cases do not validate BF16, all shapes,
all trained models or the full pipeline.

With MLX, PyTorch and the pinned upstream source available:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_layers_mlx.py' -v
```

Set `IDOL_TEST_MLX_DEVICE=gpu` to run MLX on Metal when available; PyTorch
runs on CPU. CI runs this test file on MLX CPU. See also the
[scalar sinusoidal embedding](sinusoidal-embedding.md).
