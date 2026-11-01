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
        # Equivalent to 1 + (rescale_t - 1) * t without cancellation near t=1.
        denominator = (1.0 - t) + rescale_t * t
        result.append(rescale_t * t / denominator)
    return tuple(result)


def sample_euler(model, noise, condition, *, steps=50, rescale_t=3.0,
                 neg_condition=None, cfg_strength=0.0, batch_cfg=False):
    """Integrate velocities with optional classifier-free guidance.

    `model(sample, time, condition)` receives a float32 time vector scaled by
    1000. Positive guidance requires a negative condition. Batched guidance
    assumes the model processes batch elements independently. The caller
    supplies initial noise and controls lazy MLX evaluation.
    """
    if isinstance(cfg_strength, bool) or not isinstance(cfg_strength, numbers.Real) or not math.isfinite(cfg_strength):
        raise ValueError("cfg_strength must be a finite number.")
    if not isinstance(batch_cfg, bool):
        raise ValueError("batch_cfg must be a boolean.")
    if cfg_strength > 0 and neg_condition is None:
        raise ValueError("A negative condition is required when guidance is enabled.")
    grid = time_grid(steps, rescale_t)
    sample = noise
    for t, next_t in zip(grid, grid[1:]):
        if cfg_strength > 0 and batch_cfg:
            count = sample.shape[0]
            both_sample = mx.concatenate([sample, sample], axis=0)
            both_condition = mx.concatenate([condition, neg_condition], axis=0)
            times = mx.full((both_sample.shape[0],), 1000.0 * t, dtype=mx.float32)
            both = model(both_sample, times, both_condition)
            velocity = (1 + cfg_strength) * both[:count] - cfg_strength * both[count:]
        else:
            times = mx.full((sample.shape[0],), 1000.0 * t, dtype=mx.float32)
            velocity = model(sample, times, condition)
            if cfg_strength > 0:
                negative = model(sample, times, neg_condition)
                velocity = (1 + cfg_strength) * velocity - cfg_strength * negative
        sample = sample - velocity * (t - next_t)
    return sample
