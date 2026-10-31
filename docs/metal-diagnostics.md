# MLX and Metal diagnostics

The `idol_mlx.device.require_metal()` check validates the intended Apple
Silicon macOS environment and runs a small evaluated MLX operation explicitly
on the GPU. It raises `MetalUnavailableError` with an actionable message if
the platform, MLX initialization, Metal availability or GPU operation fails.

This check does not change MLX's default device. Successful detection confirms
only that this small operation ran; it does not establish that the model fits
in memory, that inference works, or that results match PyTorch.

## Run

Install the package as described in [mlx-package.md](mlx-package.md), then run:

```sh
.venv-mlx/bin/python -c "from idol_mlx.device import require_metal; require_metal(); print('Metal check passed')"
```

If the process cannot access a GPU, try a normal local terminal on an Apple
Silicon Mac. A headless or restricted session can prevent Metal access even
when the machine has a compatible GPU. If MLX is missing or cannot initialize,
check the active Python environment and reinstall the package. On other
platforms the check rejects execution; it does not silently select a CPU or
PyTorch backend.

## Verification

Unit tests simulate supported and unsupported platforms, missing MLX or its
dependencies, initialization failure, unavailable Metal and failed GPU
evaluation. On the reference Apple Silicon Mac the real GPU check passed both
with the default device unchanged and with CPU explicitly selected as the
default. No model weights, credentials or private data are required.

Spanish: [es/metal-diagnostics.md](../es/metal-diagnostics.md).
