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
- Synthetic reference captures and tests for selected upstream components,
  plus preliminary numerical parity metrics.

Model-specific checkpoint schemas, weight conversion, encoders, the DiT,
decoders and end-to-end inference remain to be implemented and validated.

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
inference or PyTorch/MLX parity.

Future full inference requires approved access to DINOv3 and FLUX.2 dev and
compliance with their separate licenses. See
[model access and licenses](docs/model-access.md). Never commit credentials,
private images or model weights.

## Documentation

- [Upstream architecture and tensor map](docs/architecture.md)
- [Proposed MLX architecture](docs/mlx-architecture.md)
- [Configuration and checkpoint validation](docs/config-and-weights.md)
- [Synthetic reference captures](docs/reference-capture.md)
- [Parity metrics and provisional tolerances](docs/parity-metrics.md)
- [Reproducible test images](docs/test-images.md)
- [CI and SonarCloud](docs/ci-sonarcloud.md)
- [Local output-path trust boundary](docs/output-path-security.md)
