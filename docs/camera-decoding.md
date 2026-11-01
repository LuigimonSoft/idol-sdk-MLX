# Camera latent decoding

[Español](../es/camera-decoding.md)

`idol_mlx.camera.decode_camera` converts one MLX camera latent of shape `(5,)`
into the same four fields as the [pinned upstream decoder](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/camera.py):
`yaw` and `pitch` in radians, `radius`, and vertical `fov` in degrees. It
returns a dictionary of Python floats, which is JSON serializable for finite
results. Batched latents must be selected individually; incompatible shapes
and non-MLX inputs raise `ValueError`.

```python
import json
import mlx.core as mx
from idol_mlx.camera import decode_camera

camera = decode_camera(mx.array([0.0, 1.0, 0.0, 0.5, 0.25]))
camera_json = json.dumps(camera, allow_nan=False)
```

As in the reference, the first three components determine direction, component
four is inverse distance and component five is half-width. A direction norm
at or below `1e-6` falls back to `(0, 0, 1)`; inverse distance below `1e-3`
uses radius 1000 for the orthographic approximation. Values at or above the
threshold use perspective. The implementation transfers scalar results to
Python and is not a differentiable, batched or fully GPU-resident operation.

Tests compare controlled FP32 and FP16 latents against the pinned PyTorch CPU
reference, including both projection modes and threshold-adjacent directions,
at `atol=2e-6`, `rtol=2e-6`. The tests run with MLX CPU in CI and were also
run locally with Metal. They validate decoding of controlled inputs, not the
quality of predicted cameras or end-to-end inference.
