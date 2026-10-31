"""Validate that the local MLX runtime can execute on Apple Metal."""

import importlib
import platform
import sys


class MetalUnavailableError(RuntimeError):
    """MLX or its Metal GPU backend is unavailable in this environment."""


def require_metal():
    """Check Apple Silicon and execute a small MLX operation on the GPU.

    Raises MetalUnavailableError with actionable diagnostics. This does not
    change the default MLX device and does not imply model inference works.
    """
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise MetalUnavailableError("Apple Silicon macOS is required for the MLX Metal backend.")
    try:
        mx = importlib.import_module("mlx.core")
    except ModuleNotFoundError as error:
        if error.name in ("mlx", "mlx.core"):
            raise MetalUnavailableError("MLX is not installed; install idol-sdk-mlx in a compatible environment.") from error
        raise MetalUnavailableError("An MLX dependency is missing; reinstall the MLX environment.") from error
    except (ImportError, OSError, RuntimeError) as error:
        raise MetalUnavailableError(
            "MLX could not initialize; check its installation and Metal GPU access."
        ) from error
    try:
        if not mx.metal.is_available():
            raise MetalUnavailableError(
                "Metal is unavailable; run on an Apple Silicon Mac with GPU access, outside a restricted or headless environment."
            )
        value = mx.array([1.0])
        result = mx.add(value, value, stream=mx.gpu)
        mx.eval(result)
    except MetalUnavailableError:
        raise
    except Exception as error:
        raise MetalUnavailableError(
            "The MLX Metal GPU operation failed; check GPU access and the MLX installation."
        ) from error
