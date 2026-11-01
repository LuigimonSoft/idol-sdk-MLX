"""Deterministic Euler flow integration for MLX inference."""

import math
import numbers

import mlx.core as mx


def time_grid(steps, rescale_t=3.0):
    """Return the upstream shifted time grid from 1 to 0 as Python floats."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    if isinstance(rescale_t, bool) or not isinstance(rescale_t, numbers.Real):
        raise ValueError("rescale_t must be a finite positive number.")
    rescale_t = float(rescale_t)
    if not math.isfinite(rescale_t) or rescale_t <= 0:
        raise ValueError("rescale_t must be a finite positive number.")
    result = []
    for i in range(steps + 1):
        t = 1.0 - i / steps
        denominator = 1.0 + (rescale_t - 1.0) * t
        if denominator <= 0:
            raise ValueError("rescale_t is too small for stable time-grid calculation.")
        result.append(rescale_t * t / denominator)
    return tuple(result)


def sample_euler(model, noise, condition, *, steps=50, rescale_t=3.0):
    """Integrate conditional velocities; classifier-free guidance is not applied.

    `model(sample, time, condition)` receives a float32 time vector scaled by
    1000. The supplied noise is the initial state; no random draws are made.
    MLX evaluation remains lazy and is controlled by the caller.
    """
    grid = time_grid(steps, rescale_t)
    sample = noise
    for t, next_t in zip(grid, grid[1:]):
        times = mx.full((sample.shape[0],), 1000.0 * t, dtype=mx.float32)
        velocity = model(sample, times, condition)
        sample = sample - velocity * (t - next_t)
    return sample
