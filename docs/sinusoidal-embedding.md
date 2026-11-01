# Preliminary sinusoidal scalar embedding

[Español](../es/sinusoidal-embedding.md)

`idol_mlx.layers.SinusoidalEmbedder` ports the scalar time/octree-level
embedding in the [pinned upstream SDK](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/layers.py).
It constructs exponentially spaced FP32 frequencies, concatenates cosine and
sine features, optionally scales angles by `2*pi`, then applies two biased
linear layers with SiLU between them. The input is a one-dimensional batch of
scalars; scalar inputs are cast to FP32 before forming the features. The port
preserves the original layer layout and does not automatically load checkpoints.

Synthetic tests copy FP32 weights and biases from the upstream PyTorch module
and compare outputs with shared inputs, including negative and zero values,
FP32/FP16 inputs, frequency sizes 8 and 10, two periods and both `two_pi`
settings. Tolerance is `atol=1e-5, rtol=1e-4`. These tests do not validate
trained checkpoints, lower-precision weights, BF16, all numerical ranges or
full-model inference. Other temporal and spatial embeddings remain pending.

With MLX, PyTorch and the pinned source installed, run:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_layers_mlx.py' -v
```

Set `IDOL_TEST_MLX_DEVICE=gpu` for MLX Metal when available; PyTorch runs on
CPU. The existing CI shared-layer parity step runs the same tests on MLX CPU.
