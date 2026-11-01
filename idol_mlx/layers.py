"""MLX ports of selected shared layers from the upstream Idol SDK."""

import mlx.core as mx
import mlx.nn as nn


class LayerNorm32(nn.LayerNorm):
    """Normalize in FP32, then restore the input dtype, as in upstream Idol."""

    def __init__(self, dims, eps=1e-5, elementwise_affine=True):
        super().__init__(dims, eps=eps, affine=elementwise_affine)

    def __call__(self, x):
        return super().__call__(x.astype(mx.float32)).astype(x.dtype)


class MultiHeadRMSNorm(nn.Module):
    """Normalize each head using upstream's clamped L2 norm and learned scale."""

    def __init__(self, dim, heads):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = mx.ones((heads, dim))

    def __call__(self, x):
        value = x.astype(mx.float32)
        norm = mx.sqrt(mx.sum(mx.square(value), axis=-1, keepdims=True))
        normalized = value / mx.maximum(norm, 1e-12)
        return (normalized * self.gamma * self.scale).astype(x.dtype)
