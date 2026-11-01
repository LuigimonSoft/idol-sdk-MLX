# Adapted from speridlabs/idol-sdk idol/camera.py, licensed under Apache-2.0.
"""Decode a single Idol camera latent into the upstream JSON-compatible format."""

import math

import mlx.core as mx


def decode_camera(camera_latent):
    """Return yaw/pitch in radians, radius and vertical FOV in degrees.

    ``camera_latent`` is a single MLX array of shape ``(5,)``. The result is
    a dictionary of Python floats, suitable for JSON serialization.
    """
    if not isinstance(camera_latent, mx.array) or camera_latent.shape != (5,):
        raise ValueError("camera_latent must be an MLX array of shape (5,).")
    direction = camera_latent[:3].astype(mx.float32)
    inverse_distance = float(camera_latent[3].item())
    half_width = float(camera_latent[4].item())
    norm = float(mx.linalg.norm(direction).item())
    dx, dy, dz = (direction / norm).tolist() if norm > 1e-6 else (0.0, 0.0, 1.0)
    if inverse_distance < 1e-3:
        radius = 1000.0
        fov = math.degrees(2 * math.atan(half_width / radius))
    else:
        radius = 1.0 / inverse_distance
        fov = math.degrees(2 * math.atan(half_width * inverse_distance))
    # Preserve NaN behavior while clamping normal finite rounding errors.
    if dz > 1.0:
        dz = 1.0
    elif dz < -1.0:
        dz = -1.0
    return {"yaw": math.atan2(dx, dy), "pitch": math.asin(dz), "radius": radius, "fov": fov}
