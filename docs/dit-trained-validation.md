# Reduced trained-checkpoint denoiser comparison

[Español](../es/dit-trained-validation.md)

This diagnostic addresses issue #24; it does not validate general or full-size
parity. The
checkpoint `denoiser.safetensors` and configuration from
[`speridlabs/idol` revision `2731c2291fc65a797c5cb627a5e5526fc1490902`](https://huggingface.co/speridlabs/idol/tree/2731c2291fc65a797c5cb627a5e5526fc1490902)
were downloaded to a temporary local directory. The checkpoint contains 560
FP32 tensors and was loaded strictly into both the upstream PyTorch model and
MLX model using the full original configuration. No weights or input data were
committed. The reference was explicitly run in FP32, not its default mixed
precision. Both models then used only the first eight `pos_pe` positions and
identical synthetic inputs: batch 1, eight latent tokens, four condition tokens,
and time 500. This reduced sequence does not validate the full 8192-token run.

Before the canonical-frequency correction, on an Apple M4 with 32 GB unified
memory, macOS 27.2, MLX 0.32.3 and PyTorch 2.15.0, using provisional
`atol=2e-5, rtol=2e-4`:

| MLX device | Position max absolute error | Latent max absolute error | Camera max absolute error | Result |
| --- | ---: | ---: | ---: | --- |
| CPU | 0.0312334 | 0.0011998 | 0.0019016 | Fail |
| Metal GPU | 0.0312334 | 0.0020876 | 0.0022813 | Fail |

The position embedding differs before the denoiser blocks. A separate frequency
inspection found a maximum absolute frequency difference of 0.00390625 between
PyTorch CPU and MLX Metal, with differences in 56 of 170 frequency values.
Supplying the PyTorch-computed frequencies directly to MLX reduced maximum
position error from 0.0312334 to 1.78814e-7 for this input. This isolates
frequency calculation as the dominant cause of this positional discrepancy.
A second MLX forward pass now substitutes that
same reference-derived position embedding, without changing production code.
For the same reduced input and tolerances, its trained-checkpoint output was:

| MLX device | Position max absolute error | Latent max absolute error | Camera max absolute error | Shared-frequency result |
| --- | ---: | ---: | ---: | --- |
| CPU | 1.19209e-7 | 7.33137e-6 | 1.49012e-5 | Pass |
| Metal GPU | 1.78814e-7 | 5.12600e-6 | 1.10269e-5 | Pass |

The native MLX path now uses the canonical, configuration-derived FP32
frequencies documented in [spatial embeddings](spatial-embeddings.md). It
requires no PyTorch at inference. Repeating this reduced comparison after the
correction produced the same errors and passes shown in the shared-frequency
table above, on both CPU and Metal; all six diagnostic comparisons passed.
These measurements used Python 3.12 in a temporary diagnostic environment
(the package supports Python 3.13), with the same MLX and PyTorch versions.
They do not establish general parity, full-size memory requirements, or
end-to-end inference. Other frequency configurations still use the general MLX
calculation and are not covered by this trained-checkpoint result. The
provisional tolerances were not increased. Disabling `MLX_ENABLE_TF32` did not
change the original GPU output errors in this case.

## Additional reduced scenarios

The same checkpoint and environment were also checked with three deterministic
synthetic inputs on each MLX device. The tolerances were unchanged, and all six
comparisons passed for every scenario. The table gives the maximum absolute
errors for the native denoiser outputs; passing also depends on relative error.

| Device | Latent tokens | Context tokens | Seed | Timestep | Latent max error | Camera max error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 8 | 4 | 0 | 0 | 3.77465e-6 | 2.65241e-5 |
| CPU | 32 | 8 | 42 | 250 | 4.05312e-6 | 9.20892e-6 |
| CPU | 64 | 16 | 123 | 1000 | 1.97887e-5 | 1.07288e-5 |
| Metal GPU | 8 | 4 | 0 | 0 | 2.86102e-6 | 2.87294e-5 |
| Metal GPU | 32 | 8 | 42 | 250 | 4.05312e-6 | 1.44541e-5 |
| Metal GPU | 64 | 16 | 123 | 1000 | 2.49147e-5 | 1.74046e-5 |

All cases use batch 1 and crop the first positions from the checkpoint. These
are independent forwards, not a sampler trajectory. They do not validate the
8192-token model, mixed precision, other checkpoints, or end-to-end inference.
The random generator is NumPy's `default_rng`; its seed reproduces the inputs
for a fixed NumPy version, shapes, and generation order.

## Reproduce

Install MLX and the pinned reference dependencies in one environment. Obtain
both files under the model's applicable license and terms; keep them local.
From the project root, with the pinned source checkout on `PYTHONPATH`:

```sh
PYTHONPATH=../idol-sdk-reference .venv-reference/bin/python -m scripts.validate_dit_checkpoint \
  --config /trusted/path/config.json --checkpoint /trusted/path/denoiser.safetensors --device gpu
```

Use `--device cpu` for MLX CPU, `--tokens` or `--context-tokens` to select
reduced lengths, `--seed` for an unsigned 64-bit seed, and `--timestep` for a
finite value from 0 to 1000. Defaults remain seed 2026 and timestep 500. The JSON output contains only aggregate comparison metrics;
exit 0 means all six comparisons pass, exit 1 means at least one comparison
fails. The tool loads the full model/checkpoint, so it needs sufficient memory.
It does not write weights or inputs. The automated test uses a tiny synthetic
checkpoint and runs in CI; trained weights are never needed in CI. This tool
does not validate full-size memory use, encoders, decoders or image-to-3D output.
