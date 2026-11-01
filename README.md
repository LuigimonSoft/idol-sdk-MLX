# Idol SDK for MLX

An in-progress Apple Silicon port of [Idol SDK](https://github.com/speridlabs/idol-sdk),
which converts a single image into a 3D Gaussian splat.

**Status: early development.** The MLX image-to-3D pipeline, model conversion,
and PLY generation are not implemented yet. Do not expect the installation
below to perform inference.

[Español](es/README.md)

## Available now

- An installable `idol_mlx` package scaffold for the tested Python 3.13 and
  MLX 0.32.3 baseline.
- A Metal diagnostic that tests a small GPU operation without changing the
  default MLX device.
- Common model-configuration validation and a local MLX safetensors loader
  that checks an explicitly supplied tensor schema.
- Explicit in-memory tensor name mapping and standard ungrouped Conv2d layout
  conversion, with strict output-schema validation.
- An opt-in preliminary precision and explicit evaluation-boundary policy.
- Selected MLX normalization, feed-forward and sinusoidal scalar embedding
  layers plus spatial and real-valued rotary embeddings with synthetic PyTorch parity checks.
- Preliminary decoder self/cross-attention and denoiser rotary attention with
  synthetic parity checks.
- Preliminary unified and decoder transformer blocks with synthetic FP32 parity checks.
- A preliminary Euler sampler with optional classifier-free guidance and synthetic parity checks.
- Explicit seeded MLX noise generation and per-step synthetic trajectory checks.
- MLX camera-latent decoding to the upstream JSON-compatible fields.
- FP32 Gaussian attribute conversion using the upstream position, scale, rotation, opacity and SH conventions.
- A preliminary FP32 Gaussian feature decoder with strict checkpoint mapping;
  some reduced trained-weight Metal cases still fail provisional tolerances;
  optional diagnostics compare block outputs and attention implementations.
- Preliminary octree traversal and anchor sampling with controlled MLX randomness.
- A systematic octree resampling primitive with explicit random offsets.
- A preliminary FP32 octree probability network and strict checkpoint mapping,
  tested against synthetic reference weights and reduced trained-checkpoint scenarios.
- A preliminary FP32 denoiser and strict checkpoint mapping. A reduced
  trained-weight diagnostic passes provisional tolerances for several reduced FP32 inputs.
- A preliminary file-based parity comparison tool with explicit tolerances.
- Synthetic reference captures and tests for selected upstream components,
  plus preliminary numerical parity metrics.

Preliminary FP32 denoiser and decoder paths have limited synthetic and
trained-checkpoint validation. Additional checkpoint schemas and conversion,
encoders, complete decoders and end-to-end inference remain work in progress.

## Installation

On Apple Silicon, with CPython 3.13.14 and
[uv](https://docs.astral.sh/uv/) available:

```sh
uv venv --python 3.13.14 .venv-mlx
uv pip install --python .venv-mlx/bin/python -e .
.venv-mlx/bin/python -c "import idol_mlx; print(idol_mlx.__version__)"
```

Check the Metal runtime separately:

```sh
.venv-mlx/bin/python -c "from idol_mlx.device import require_metal; require_metal(); print('Metal check passed')"
```

See [MLX package installation](docs/mlx-package.md) and
[Metal diagnostics](docs/metal-diagnostics.md). Metal may be inaccessible in
restricted or headless sessions even on compatible hardware.

## Tests and model access

The full unit-test suite uses a separate reference environment and an upstream
checkout pinned to a specific commit. Follow
[development environments](docs/environments.md) and
[reference component checks](docs/reference-components.md) before running:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference .venv-reference/bin/python -m unittest discover -s tests -v
```

If `IDOL_REFERENCE_SOURCE` is not set, reference-dependent tests are skipped;
a skipped run is not full validation. The current tests do not prove full-model
inference or PyTorch/MLX parity. A [reduced trained-checkpoint diagnostic](docs/dit-trained-validation.md)
compares denoiser outputs with both native and reference-derived position
frequencies; both paths pass for the recorded reduced FP32 scenarios.

Future full inference requires approved access to DINOv3 and FLUX.2 dev and
compliance with their separate licenses. See
[model access and licenses](docs/model-access.md). Never commit credentials,
private images or model weights.

## Documentation

- [Gaussian attribute conversion](docs/gaussian-attributes.md)

- [Gaussian feature decoder](docs/gaussian-decoder.md)

- [Octree limits and memory observations](docs/octree-profiling.md)

- [Octree anchor sampling](docs/octree-traversal.md)

- [Octree systematic resampling](docs/octree-resampling.md)

- [Octree probability network](docs/octree-network.md)

- [Camera decoding](docs/camera-decoding.md)

- [Upstream architecture and tensor map](docs/architecture.md)
- [Proposed MLX architecture](docs/mlx-architecture.md)
- [Configuration and checkpoint validation](docs/config-and-weights.md)
- [Preliminary weight mapping](docs/weight-mapping.md)
- [Precision and lazy evaluation policy](docs/precision-evaluation.md)
- [MLX normalization layers](docs/normalization-layers.md)
- [MLX feed-forward layers](docs/feedforward-layers.md)
- [Sinusoidal scalar embedding](docs/sinusoidal-embedding.md)
- [Spatial embeddings](docs/spatial-embeddings.md)
- [Rotary and RePo 3D](docs/rotary-repo.md)
- [Decoder attention](docs/decoder-attention.md)
- [Denoiser rotary attention](docs/rotary-attention.md)
- [Unified transformer block](docs/unified-transformer.md)
- [Decoder transformer blocks](docs/decoder-transformers.md)
- [Euler flow sampler](docs/euler-sampler.md)
- [Seeded noise and reproducibility](docs/reproducibility.md)
- [Synthetic sampler trajectories](docs/sampler-trajectories.md)
- [Preliminary FP32 denoiser](docs/dit-fp32.md)
- [FP32 denoiser checkpoint loading](docs/dit-checkpoints.md)
- [Trained-checkpoint diagnostic](docs/dit-trained-validation.md)
- [Synthetic reference captures](docs/reference-capture.md)
- [Parity metrics and provisional tolerances](docs/parity-metrics.md)
- [Comparing backend outputs](docs/parity-harness.md)
- [Reproducible test images](docs/test-images.md)
- [CI and SonarCloud](docs/ci-sonarcloud.md)
- [Local output-path trust boundary](docs/output-path-security.md)
