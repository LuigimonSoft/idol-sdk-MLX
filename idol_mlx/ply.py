# Adapted from speridlabs/idol-sdk idol/gaussian.py, Apache-2.0.
"""Export activated MLX Gaussian attributes as binary 3DGS PLY."""

import math
from pathlib import Path

import mlx.core as mx


_FIELDS = ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
           "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")


def save_gaussian_ply(path, attributes, *, batch_index=0):
    """Write one batch item as a new binary little-endian 3DGS PLY file.

    ``attributes`` contains batched activated FP32 arrays returned by
    ``decode_gaussian_attributes``. The local output path is trusted. Existing
    files are never replaced; inputs are validated before opening the file.
    """
    if not isinstance(attributes, dict):
        raise ValueError("attributes must be a dictionary of batched MLX arrays.")
    names = ("xyz", "features_dc", "scaling", "rotation", "opacity")
    if any(name not in attributes for name in names):
        raise ValueError("attributes must contain xyz, features_dc, scaling, rotation and opacity.")
    values = {name: attributes[name] for name in names}
    if any(not isinstance(value, mx.array) or value.dtype != mx.float32 for value in values.values()):
        raise ValueError("all Gaussian attributes must be float32 MLX arrays.")
    xyz = values["xyz"]
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or xyz.shape[0] < 1:
        raise ValueError("xyz must have shape (batch, count, 3) with a nonempty batch.")
    batch, count, _ = xyz.shape
    expected = {"features_dc": (batch, count, 1, 3), "scaling": (batch, count, 3),
                "rotation": (batch, count, 4), "opacity": (batch, count, 1)}
    if any(values[name].shape != shape for name, shape in expected.items()):
        raise ValueError("Gaussian attribute shapes must match xyz.")
    if isinstance(batch_index, bool) or not isinstance(batch_index, int) or not 0 <= batch_index < batch:
        raise ValueError("batch_index must select an existing batch item.")
    output_path = Path(path)
    if output_path.suffix.lower() != ".ply":
        raise ValueError("output path must end in .ply.")
    selected = {name: value[batch_index] for name, value in values.items()}
    if not all(bool(mx.all(mx.isfinite(value)).item()) for value in selected.values()):
        raise ValueError("Gaussian attributes must be finite.")

    positions = selected["xyz"]
    positions = mx.stack((positions[:, 0], -positions[:, 2], positions[:, 1]), axis=-1)
    color = selected["features_dc"].reshape(count, 3)
    opacity = mx.clip(selected["opacity"], 1e-6, 1 - 1e-6)
    opacity = mx.log(opacity / (1 - opacity))
    scaling = mx.log(mx.maximum(selected["scaling"], 1e-6))
    w, x, y, z = (selected["rotation"][:, i] for i in range(4))
    half_rotation = math.sqrt(0.5)
    rotation = mx.stack((half_rotation * (w - x), half_rotation * (x + w),
                         half_rotation * (y - z), half_rotation * (z + y)), axis=-1)
    rows = mx.contiguous(mx.concatenate((positions, mx.zeros_like(positions), color, opacity, scaling, rotation), axis=1))
    mx.eval(rows)
    if not bool(mx.all(mx.isfinite(rows)).item()):
        raise ValueError("transformed Gaussian attributes must be finite.")
    header = ("ply\nformat binary_little_endian 1.0\n" + f"element vertex {count}\n" +
              "".join(f"property float {name}\n" for name in _FIELDS) + "end_header\n").encode("ascii")
    with output_path.open("xb") as stream:
        stream.write(header)
        if count:
            stream.write(memoryview(rows).cast("B"))
