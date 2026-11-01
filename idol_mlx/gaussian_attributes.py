# Adapted from speridlabs/idol-sdk idol/models/decoder.py and idol/gaussian.py, Apache-2.0.
"""Convert Gaussian decoder features into batched 3DGS attributes."""

import math
import numbers

import mlx.core as mx
import mlx.nn as nn


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return float(value)


def decode_gaussian_attributes(points, features, *, representation_config,
                               points_offset_perturbation, base_offset_scale):
    """Return batched z-up xyz, SH color, scale, quaternion and opacity arrays.

    Inputs are FP32 MLX arrays and the buffers from a loaded Gaussian decoder.
    Each result has leading dimensions ``(batch, anchors * gaussians)``; color
    additionally has the reference's degree-zero SH dimension of size one.
    """
    if not isinstance(representation_config, dict):
        raise ValueError("representation_config must be a dictionary.")
    per_anchor = representation_config.get("num_gaussians")
    if isinstance(per_anchor, bool) or not isinstance(per_anchor, int) or per_anchor < 1:
        raise ValueError("num_gaussians must be a positive integer.")
    arrays = (points, features, points_offset_perturbation, base_offset_scale)
    if any(not isinstance(value, mx.array) or value.dtype != mx.float32 for value in arrays):
        raise ValueError("points, features and buffers must be float32 MLX arrays.")
    if points.ndim != 3 or points.shape[-1] != 3 or any(size < 1 for size in points.shape[:2]):
        raise ValueError("points must have non-empty shape (batch, anchors, 3).")
    if features.shape != (*points.shape[:2], per_anchor * 15):
        raise ValueError("features must have shape (batch, anchors, num_gaussians * 15).")
    if points_offset_perturbation.shape != (per_anchor, 3) or base_offset_scale.shape != ():
        raise ValueError("decoder buffer shapes do not match num_gaussians.")
    rates = representation_config.get("lr")
    if not isinstance(rates, dict):
        raise ValueError("lr must be a dictionary.")
    names = ("_xyz", "_features_dc", "_scaling", "_rotation", "_opacity")
    rates = {name: _finite_number(rates.get(name), f"lr.{name}") for name in names}
    size = _finite_number(representation_config.get("perturbe_size"), "perturbe_size")
    scale_bias = _finite_number(representation_config.get("scaling_bias"), "scaling_bias")
    opacity_bias = _finite_number(representation_config.get("opacity_bias"), "opacity_bias")
    kernel = _finite_number(representation_config.get("3d_filter_kernel_size"), "3d_filter_kernel_size")
    if scale_bias <= 0 or not 0 < opacity_bias < 1 or kernel < 0:
        raise ValueError("scaling_bias must be positive, opacity_bias in (0, 1), and kernel non-negative.")

    batch, anchors, _ = points.shape
    start = 0
    parts = {}
    for name, dim in (("_xyz", 3), ("_features_dc", 3), ("_scaling", 3),
                      ("_rotation", 4), ("_opacity", 1), ("_offset_scale", 1)):
        end = start + per_anchor * dim
        parts[name] = features[..., start:end].reshape(batch, anchors, per_anchor, dim)
        start = end
    offset_scale = nn.softplus(parts["_offset_scale"] + base_offset_scale)
    offset = parts["_xyz"] * rates["_xyz"] + points_offset_perturbation
    xyz = (points[:, :, None, :] + mx.tanh(offset) * (0.5 * size) * offset_scale - 0.5).reshape(batch, -1, 3)
    color = (parts["_features_dc"] * rates["_features_dc"]).reshape(batch, -1, 1, 3)
    scale_input = parts["_scaling"] * rates["_scaling"]
    scaling_bias = mx.array(scale_bias, dtype=mx.float32)
    scale_offset = scaling_bias + mx.log(-mx.expm1(-scaling_bias))
    scale = nn.softplus(scale_input + scale_offset)
    scale = mx.sqrt(mx.square(scale) + kernel**2).reshape(batch, -1, 3)
    rotation = parts["_rotation"] * rates["_rotation"] + mx.array([1.0, 0.0, 0.0, 0.0])
    norm = mx.sqrt(mx.sum(mx.square(rotation), axis=-1, keepdims=True))
    rotation = (rotation / mx.maximum(norm, 1e-12)).reshape(batch, -1, 4)
    opacity_bias_value = mx.array(opacity_bias, dtype=mx.float32)
    opacity_offset = mx.log(opacity_bias_value / (1 - opacity_bias_value))
    opacity = mx.sigmoid(parts["_opacity"] * rates["_opacity"] + opacity_offset).reshape(batch, -1, 1)
    return {"xyz": xyz, "features_dc": color, "scaling": scale, "rotation": rotation, "opacity": opacity}
