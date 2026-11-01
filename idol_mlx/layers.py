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


class FeedForwardNet(nn.Module):
    """Two biased linear layers with the upstream tanh-approximate GELU."""

    def __init__(self, channels, mlp_ratio=4.0):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.gelu_approx,
                                 nn.Linear(hidden, channels))

    def __call__(self, x):
        return self.mlp(x)


class MLP(nn.Module):
    """Upstream MLP layout with tanh-approximate GELU between linear layers."""

    def __init__(self, channels, inner_channels, channels_out, mlp_layer_num=2):
        super().__init__()
        layers = []
        for i in range(mlp_layer_num - 1):
            layers.append(nn.Linear(channels if i == 0 else inner_channels, inner_channels))
            layers.append(nn.gelu_approx)
        layers.append(nn.Linear(inner_channels, channels_out))
        self.mlp = nn.Sequential(*layers)

    def __call__(self, x):
        return self.mlp(x)
