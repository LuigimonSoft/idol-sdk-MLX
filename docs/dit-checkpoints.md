# Preliminary FP32 denoiser checkpoint loading

[Español](../es/dit-checkpoints.md)

`idol_mlx.dit_weights.load_dit_checkpoint(model, path)` loads a **trusted local**
`safetensors` file with names from the pinned upstream PyTorch denoiser. It maps
the known sequential-container names to MLX, validates the complete source
and target schemas against the supplied `LatentSeqMMFlowModel`, and then loads
parameters and the `pos_pe` buffer strictly. No tensors are cast, transposed,
ignored or silently accepted with wrong shape or dtype. All model and checkpoint
tensors must be FP32. Validate that the model configuration matches the intended
checkpoint before loading.

```python
from idol_mlx.dit_weights import load_dit_checkpoint

# model is an already configured LatentSeqMMFlowModel.
# This path is a trusted local safetensors checkpoint matching the model.
load_dit_checkpoint(model, "/trusted/path/denoiser.safetensors")
```

The loader has been tested with a synthetic small FP32 checkpoint containing all
123 expected tensors, not with trained checkpoints. Tests compare the loaded
model's forward output with PyTorch and confirm missing/extra keys, wrong shape,
wrong dtype and non-FP32 targets are rejected. The tests check that rejected
checkpoints do not change model parameters. Upstream mixed-precision checkpoints
are not supported by this loader without a separately validated conversion.
Full-size memory requirements, trained model outputs and end-to-end inference
remain unvalidated. No model files, credentials or private inputs belong in the
repository.

See [preliminary FP32 denoiser](dit-fp32.md) for test commands and limitations.
