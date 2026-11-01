# Preliminary octree probability network

[Español](../es/octree-network.md)

`idol_mlx.octree.OctreeProbabilityFixedlenDecoder` implements the FP32 forward
path of the [pinned upstream octree decoder](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/decoder.py).
It accepts cell centers `[B, L, 3]`, level `[B]`, and conditioning features
`[B, Lc, cond_channels]`, and returns logits `[B, L, 8]`. Apply
`mx.softmax(logits, axis=-1)` for child probabilities. Sampling and variable-size
octree traversal are not implemented here.

`idol_mlx.octree_weights.load_octree_checkpoint` reads a trusted local
upstream-named FP32 safetensors file. It checks the full source schema, target
schema, shapes, and dtypes before updating the model; it does not cast, reshape,
or ignore tensors. No weights are included or downloaded by this API.

Synthetic tests compare logits and probabilities with shared PyTorch weights,
multiple batches and lengths, at provisional `atol=2e-5`, `rtol=2e-4`. The
reference is explicitly converted to FP32; its usual mixed-precision behavior is
not validated. Tests also check that invalid checkpoints do not mutate the
model. CI executes them on MLX CPU; they have also been run locally on Metal.
This preliminary network is not a full octree sampler or an end-to-end decoder.

## Reduced trained-checkpoint diagnostic

The strictly loaded `decoder.safetensors` (60 FP32 tensors) from
[`speridlabs/idol` revision `2731c2291fc65a797c5cb627a5e5526fc1490902`](https://huggingface.co/speridlabs/idol/tree/2731c2291fc65a797c5cb627a5e5526fc1490902)
was compared with the reference explicitly run in FP32. On an Apple M4, MLX
0.32.3, PyTorch 2.15.0 and Python 3.12. The same results were reproduced
with supported Python 3.13.14. Using batch 1, 8 synthetic cells,
4 condition tokens, seed 2026 and level 4, the native implementation before
the canonical-frequency correction **did not pass** provisional tolerances
`atol=2e-5, rtol=2e-4`:

| MLX device | Position max error | Logits max error (failed / 64) | Probability max error (failed / 64) |
| --- | ---: | ---: | ---: |
| CPU | 0.00189202 | 0.00456715 (12) | 0.000149608 (5) |
| Metal GPU | 0.00165553 | 0.00364113 (12) | 0.000131965 (5) |

The position embedding differed before the decoder blocks. Supplying the
reference frequencies reduced the embedding error and made the original
reduced comparison pass. The native MLX implementation now uses those
configuration-derived canonical FP32 frequencies for the official decoder
schedule. With the correction, Python 3.13.14, the same checkpoint and
provisional tolerances, all three metrics passed for each scenario below:

| Device | Cells | Context tokens | Seed | Level | Logits max error | Probability max error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 8 | 4 | 2026 | 4 | 2.38419e-5 | 1.90735e-6 |
| CPU | 16 | 7 | 42 | 2 | 1.71661e-5 | 1.65403e-6 |
| CPU | 32 | 8 | 123 | 8 | 3.09944e-5 | 5.69224e-6 |
| Metal GPU | 8 | 4 | 2026 | 4 | 2.52724e-5 | 3.21865e-6 |
| Metal GPU | 16 | 7 | 42 | 2 | 2.92659e-5 | 2.68221e-6 |
| Metal GPU | 32 | 8 | 123 | 8 | 5.91278e-5 | 7.09295e-6 |

Passing depends on both absolute and relative error. Position maximum error was
1.19209e-7 on CPU and 1.78814e-7 on Metal in all six runs. These are independent
FP32 forward passes with batch 1 and synthetic inputs; they do not validate
mixed precision, full octree traversal, the Gaussian decoder or end-to-end
inference. No trained weights or inputs are included in the repository.

To reproduce with the pinned upstream checkout on `PYTHONPATH` and trusted
local files, run the command below; use `--device cpu` for MLX CPU. Exit 0 means
all position, logits and probability comparisons passed; exit 1 means at least
one failed. It emits aggregate metrics only and loads the full network weights.

```sh
PYTHONPATH=../idol-sdk-reference .venv-reference/bin/python -m scripts.validate_octree_checkpoint \
  --config /trusted/path/config.json --checkpoint /trusted/path/decoder.safetensors --device gpu
```
