# Reduced trained-checkpoint denoiser comparison

[Español](../es/dit-trained-validation.md)

This is a diagnostic for issue #24, **not a passed parity validation**. The
checkpoint `denoiser.safetensors` and configuration from
[`speridlabs/idol` revision `2731c2291fc65a797c5cb627a5e5526fc1490902`](https://huggingface.co/speridlabs/idol/tree/2731c2291fc65a797c5cb627a5e5526fc1490902)
were downloaded to a temporary local directory. The checkpoint contains 560
FP32 tensors and was loaded strictly into both the upstream PyTorch model and
MLX model using the full original configuration. No weights or input data were
committed. The reference was explicitly run in FP32, not its default mixed
precision. Both models then used only the first eight `pos_pe` positions and
identical synthetic inputs: batch 1, eight latent tokens, four condition tokens,
and time 500. This reduced sequence does not validate the full 8192-token run.

On an Apple M4 with 32 GB unified memory, macOS 27.2, MLX 0.32.3 and PyTorch
2.15.0, using provisional `atol=2e-5, rtol=2e-4`:

| MLX device | Position max absolute error | Latent max absolute error | Camera max absolute error | Result |
| --- | ---: | ---: | ---: | --- |
| CPU | 0.0312334 | 0.0011998 | 0.0019016 | Fail |
| Metal GPU | 0.0312334 | 0.0020876 | 0.0022813 | Fail |

The position embedding differs before the denoiser blocks. A separate frequency
inspection found a maximum absolute frequency difference of 0.00390625 between
PyTorch CPU and MLX Metal, with differences in 56 of 170 frequency values.
This is a plausible contributor to position error, not proof that it is the only
source. Disabling `MLX_ENABLE_TF32` did not change the observed GPU output errors
in this case. Investigating and correcting numerical differences remains work;
tolerances have not been increased to make the result pass.

## Reproduce

Install MLX and the pinned reference dependencies in one environment. Obtain
both files under the model's applicable license and terms; keep them local.
From the project root, with the pinned source checkout on `PYTHONPATH`:

```sh
PYTHONPATH=../idol-sdk-reference .venv-reference/bin/python -m scripts.validate_dit_checkpoint \
  --config /trusted/path/config.json --checkpoint /trusted/path/denoiser.safetensors --device gpu
```

Use `--device cpu` for MLX CPU, and `--tokens` or `--context-tokens` to select
reduced lengths. The JSON output contains only aggregate comparison metrics;
exit 0 means all three comparisons pass, exit 1 means at least one comparison
fails. The tool loads the full model/checkpoint, so it needs sufficient memory.
It does not write weights or inputs. The automated test uses a tiny synthetic
checkpoint and runs in CI; trained weights are never needed in CI. This tool
does not validate full-size memory use, encoders, decoders or image-to-3D output.
