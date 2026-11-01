# Gaussian PLY export

[Español](../es/gaussian-ply.md)

`idol_mlx.ply.save_gaussian_ply` exports one batch item from the activated FP32
arrays returned by [Gaussian attribute conversion](gaussian-attributes.md):

```python
from idol_mlx.ply import save_gaussian_ply

# attributes is the batched dictionary from decode_gaussian_attributes.
save_gaussian_ply("/trusted/output/object.ply", attributes, batch_index=0)
```

The path is selected by a trusted local caller and must end in `.ply`. The
function creates a new file exclusively; an existing file is never replaced.
It validates shapes and finite input and transformed values before opening the
output. It selects only the specified batch item (default 0); export batch
items separately. An I/O failure after the file is opened may leave an
incomplete file. Empty Gaussian sets produce a zero-vertex PLY.

The binary little-endian PLY has the upstream vertex properties in order:
`x`, `y`, `z`, `nx`, `ny`, `nz`, `f_dc_0`–`f_dc_2`, `opacity`,
`scale_0`–`scale_2` and `rot_0`–`rot_3`. Positions and quaternions are rotated
from z-up to y-up, normals are zero, color remains degree-zero spherical
harmonics, opacity is clamped to `[1e-6, 1 - 1e-6]` and converted to logits,
and scales are clamped to `1e-6` and converted to logarithms. Rotations use
`(w, x, y, z)` order. These conventions match the pinned upstream exporter;
no RGB conversion or additional quaternion normalization is applied.

Automated tests compare every field with the upstream PyTorch exporter for
synthetic data on CPU and Metal, including the attribute-conversion-to-PLY path,
axis rotation, clamping, empty output and exclusive creation. A 1500-Gaussian
synthetic colored sphere exported by this function was also opened and rendered
locally in the [antimatter15/splat viewer](https://github.com/antimatter15/splat/tree/ba182b51b7c2ad5738cdd6741cd63336d27470fb)
at commit `ba182b51b7c2ad5738cdd6741cd63336d27470fb`, served on localhost in
the in-app browser. The viewer read the exported PLY directly; no user data or
weights were uploaded. This confirms one viewer and synthetic input, not
compatibility with every viewer or end-to-end image-to-3D inference. Full
Gaussian count and memory limits remain separate work in issue #36.
