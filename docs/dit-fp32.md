# Preliminary FP32 denoiser

[Español](../es/dit-fp32.md)

`idol_mlx.dit.LatentSeqMMFlowModel` implements a preliminary FP32 forward path
adapted from the [pinned upstream denoiser](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/dit.py).
It combines time and position embeddings, noise/context refiners, learned real
RePo rotations, joint transformer blocks and latent/camera output heads. Inputs
are dictionaries with `latent`, `camera`, `feature1` and `feature2` arrays;
outputs contain `latent` and `camera`. The positional buffer `pos_pe` is part
of the model state and must be populated with the appropriate positions.

This implementation is **FP32 only for the tested configuration**. The upstream
default uses mixed FP16/FP32. The synthetic reference test explicitly converts
its PyTorch model to FP32 and sets its model dtype to FP32; it does not validate
the upstream default precision path. `model_channels` must equal
`num_heads * num_head_channels`.

A tiny synthetic model test maps all 123 PyTorch parameters and buffers by name,
checks every shape and loads them strictly into MLX. It compares latent and
camera outputs for two batch/condition lengths on CPU or Metal using tolerance
`atol=2e-5, rtol=2e-4`. The preliminary [checkpoint loader](dit-checkpoints.md) provides this mapping
and strict validation for FP32 safetensors. Trained weights, upstream mixed
precision, full-size memory requirements and end-to-end inference remain unvalidated. Do not commit
model weights, private inputs or credentials.

With MLX, PyTorch and the pinned source installed:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_dit_mlx.py' -v
```

Set `IDOL_TEST_MLX_DEVICE=gpu` for MLX Metal when available; PyTorch stays on
CPU. CI runs this synthetic test on MLX CPU. The upstream source attributes its
architecture to the Wan and Hugging Face teams under Apache-2.0; the port retains
that notice in the source file.
