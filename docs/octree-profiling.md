# Octree limits and memory observations

[Español](../es/octree-profiling.md)

The preliminary FP32 octree sampler was measured on an Apple M4 with 32 GiB
unified memory, macOS 27.2, Python 3.13.14, MLX 0.32.3 and Metal. The strictly
loaded `decoder.safetensors` (60 FP32 tensors) and configuration came from
[`speridlabs/idol` revision `2731c2291fc65a797c5cb627a5e5526fc1490902`](https://huggingface.co/speridlabs/idol/tree/2731c2291fc65a797c5cb627a5e5526fc1490902).
Inputs were synthetic FP32, batch 1, seed 101 and 8192 condition tokens. Every
row is a single cold-process run; times are not a statistical benchmark.

| Anchors | Levels | Time (s) | MLX peak active (MiB) | MLX cache after (MiB) | Process peak RSS (MiB) | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 8 | 0.00012 | 194.3 | <0.1 | 246.3 | Pass; no model forward |
| 1 | 8 | 0.219 | 708.6 | 514.4 | 253.5 | Pass |
| 64 | 4 | 0.251 | 733.1 | 544.8 | 254.2 | Pass |
| 1024 | 8 | 0.613 | 986.8 | 1551.8 | 255.4 | Pass |
| 8192 | 8 | 1.800 | 1355.0 | 3486.8 | 256.2 | Pass |
| 32768 | 8 | 4.157 | 1833.4 | 5840.9 | 257.7 | Pass |

A pass means the requested output shape, finite coordinates and coordinates
within `[0, 1]` were checked. It does not measure output quality. The upstream
pipeline defaults to 8192 anchors and documents an upper range corresponding to
32768 anchors; these measurements cover the octree stage only, not Gaussian
decoding or image-to-3D generation.

Timing starts after loading and evaluating the model and condition, and includes
sampling, eager synchronization and first-use compilation for that process;
output validation is outside the timer. MLX peak active bytes are measured
since a reset immediately before sampling and include the approximately 194.3
MiB active model/input baseline, but **exclude cached buffers**. Cache after is
a separate end-of-run snapshot, not a peak; it must not be added to peak active
as if they were measured simultaneously. Process peak RSS is the macOS process
high-water mark, includes setup, and does not represent total Metal/unified
memory. These counters are not a complete system-memory measurement and may
change with input, allocator state, other workloads or hardware.

For numerical precision, a separate controlled-randomness reference comparison
used the same checkpoint, seed 101, 32 anchors, 4 levels and 32 condition tokens.
All 96 FP32 coordinates matched PyTorch exactly on both MLX CPU and Metal when
all resampling offsets and leaf jitter were fixed to `0.375`. This is one
reduced case, not an accuracy guarantee for full-size or stochastic inference.

## Reproduce

Obtain the weights under their applicable terms and keep them local. Each
profiling command starts a fresh process and writes aggregate JSON only:

```sh
.venv-reference/bin/python -m scripts.profile_octree \
  --config /trusted/path/config.json --checkpoint /trusted/path/decoder.safetensors \
  --num-points 8192 --level 8 --context-tokens 8192 --device gpu
```

For the precision case, use the pinned upstream source on `PYTHONPATH`:

```sh
PYTHONPATH=../idol-sdk-reference .venv-reference/bin/python -m scripts.validate_octree_traversal \
  --config /trusted/path/config.json --checkpoint /trusted/path/decoder.safetensors \
  --num-points 32 --level 4 --context-tokens 32 --device gpu
```

Neither command publishes weights or inputs. Results are specific to this
hardware and setup; memory limits on other machines and mixed-precision behavior
remain unvalidated.
