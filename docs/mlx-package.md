# MLX package installation

The `idol_mlx` package is an initial packaging scaffold. It does not yet
implement the inference pipeline, model loading, a CLI, or image-to-3D output.

## Supported baseline

The initial tested baseline is Apple Silicon, CPython 3.13.14 and MLX 0.32.3.
The project currently constrains Python to the 3.13 series and pins MLX 0.32.3;
other combinations need separate validation. Metal availability and device
diagnostics are addressed in issue #12. This package's installation alone does
not prove that a machine can access a GPU.

## Install from a checkout

From the repository root, with [uv](https://docs.astral.sh/uv/) and CPython
3.13.14 available:

```sh
uv venv --python 3.13.14 .venv-mlx
uv pip install --python .venv-mlx/bin/python -e .
.venv-mlx/bin/python -c "import idol_mlx; print(idol_mlx.__version__)"
```

An editable install resolves project dependencies from the configured package
index. For the separately documented hash-verified MLX dependency baseline,
see [environments.md](environments.md); that lock does not itself install
this project.

The MLX runtime dependency does not pull in PyTorch; the PyTorch reference
environment is separate. No model weights or credentials are bundled. Do not
add tokens, private images or generated weights to this repository.

## Verification

A wheel was built and installed into a fresh virtual environment on the
reference Apple Silicon Mac using CPython 3.13.14. From outside the source
checkout, package import and a simple MLX Metal computation succeeded without
importing PyTorch. This is a packaging smoke test, not model inference or
numerical parity validation.

Spanish: [es/mlx-package.md](../es/mlx-package.md).
