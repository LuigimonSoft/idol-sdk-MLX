"""MLX ports of selected shared layers from the upstream Idol SDK."""

import math

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


class SinusoidalEmbedder(nn.Module):
    """Embed scalar time or octree level using upstream frequencies and SiLU."""

    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=10000, two_pi=False):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size), nn.silu,
                                 nn.Linear(hidden_size, hidden_size))
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.two_pi = two_pi

    def __call__(self, t):
        half = self.frequency_embedding_size // 2
        freqs = mx.exp(-math.log(self.max_period) * mx.arange(half, dtype=mx.float32) / half)
        args = t[:, None].astype(mx.float32) * freqs[None, :]
        if self.two_pi:
            args = args * (2 * math.pi)
        return self.mlp(mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1))


class PcdAbsolutePositionEmbedder(nn.Module):
    """Encode per-token coordinates using the denoiser's frequency schedule."""

    def __init__(self, channels, in_channels=3, max_res=16):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.max_res = max_res
        self.freq_dim = channels // in_channels // 2

    def __call__(self, x):
        exponents = mx.arange(self.max_res, dtype=mx.float32)
        remaining = max(0, self.freq_dim - self.max_res)
        if remaining:
            tail = mx.arange(remaining, dtype=mx.float32) / remaining * self.max_res
            exponents = mx.concatenate([exponents, tail])
        frequencies = mx.power(2.0, exponents[:self.freq_dim])
        angles = mx.outer(x.reshape(-1), frequencies) * 2 * math.pi
        embedding = mx.concatenate([mx.sin(angles), mx.cos(angles)], axis=-1)
        embedding = embedding.reshape(*x.shape[:-1], -1)
        if embedding.shape[-1] < self.channels:
            padding = mx.zeros((*x.shape[:-1], self.channels - embedding.shape[-1]), dtype=mx.float32)
            embedding = mx.concatenate([embedding, padding], axis=-1)
        return embedding


class PcdAbsolutePositionEmbedderv2(nn.Module):
    """Encode per-sample coordinates using the decoder's frequency schedule."""

    def __init__(self, channels, in_channels=3, max_res=10):
        super().__init__()
        self.channels = channels
        self.freq_dim = channels // in_channels // 2
        self.max_res = max_res

    def __call__(self, x):
        logs = mx.linspace(0.0, float(self.max_res), self.freq_dim, dtype=mx.float32)
        frequencies = mx.power(2.0, logs).astype(x.dtype)
        product = mx.expand_dims(x, -1) * frequencies
        # PyTorch applies the scalar in FP32 before rounding a reduced-precision result.
        if x.dtype in (mx.float16, mx.bfloat16):
            angles = (product.astype(mx.float32) * math.pi).astype(x.dtype)
        else:
            angles = product * math.pi
        embedding = mx.concatenate([mx.sin(angles), mx.cos(angles)], axis=-1).reshape(x.shape[0], -1)
        if embedding.shape[1] < self.channels:
            padding = mx.zeros((x.shape[0], self.channels - embedding.shape[1]), dtype=embedding.dtype)
            embedding = mx.concatenate([embedding, padding], axis=-1)
        return embedding


def apply_rotary_emb(hidden_states, freqs):
    """Rotate adjacent feature pairs using real (cosine, sine) frequencies."""
    pairs = hidden_states.astype(mx.float32).reshape(*hidden_states.shape[:-1], -1, 2)
    x0, x1 = pairs[..., 0], pairs[..., 1]
    cosine, sine = freqs[..., 0], freqs[..., 1]
    rotated = mx.stack([x0 * cosine - x1 * sine, x0 * sine + x1 * cosine], axis=-1)
    return rotated.reshape(*rotated.shape[:-2], -1).astype(hidden_states.dtype)


def _clamp_mul(x, frequency):
    bounded = mx.tanh(frequency)
    return x * bounded + mx.stop_gradient(x) * (frequency - bounded)


class RePo3DRotaryEmbedding(nn.Module):
    """Predict real-valued 3D rotary frequencies from token features."""

    def __init__(self, model_channels, num_heads, head_dim, repo_hidden_ratio=0.125, max_freq=16.0):
        super().__init__()
        self.num_heads = num_heads
        hidden = int(model_channels * repo_hidden_ratio)
        self.norm = LayerNorm32(model_channels)
        self.gate_map = nn.Linear(model_channels, hidden, bias=False)
        self.content_map = nn.Linear(model_channels, hidden, bias=False)
        self.final_map = nn.Linear(hidden, 3 * num_heads, bias=False)
        first = 2 * (head_dim // 6)
        dims = (first, first, head_dim - 2 * first)
        self.freqs_0 = mx.linspace(1.0, float(max_freq), dims[0] // 2, dtype=mx.float32)
        self.freqs_1 = mx.linspace(1.0, float(max_freq), dims[1] // 2, dtype=mx.float32)
        self.freqs_2 = mx.linspace(1.0, float(max_freq), dims[2] // 2, dtype=mx.float32)

    def __call__(self, hidden_states):
        h = self.norm(hidden_states.astype(mx.float32))
        delta = self.final_map(nn.silu(self.gate_map(h)) * self.content_map(h))
        delta = delta.reshape(hidden_states.shape[0], hidden_states.shape[1], self.num_heads, 3)
        angles = [_clamp_mul(delta[..., i:i + 1], freq) * math.pi
                  for i, freq in enumerate((self.freqs_0, self.freqs_1, self.freqs_2))]
        angles = mx.concatenate(angles, axis=-1)
        return mx.stack([mx.cos(angles), mx.sin(angles)], axis=-1)


class MultiHeadAttention(nn.Module):
    """Decoder self/cross attention with upstream query/key normalization."""

    def __init__(self, channels, num_heads, ctx_channels=None, cross=False):
        super().__init__()
        self.num_heads = num_heads
        self.cross = cross
        head_dim = channels // num_heads
        if cross:
            self.to_q = nn.Linear(channels, channels)
            self.to_kv = nn.Linear(ctx_channels, channels * 2)
        else:
            self.to_qkv = nn.Linear(channels, channels * 3)
        self.q_rms_norm = MultiHeadRMSNorm(head_dim, num_heads)
        self.k_rms_norm = MultiHeadRMSNorm(head_dim, num_heads)
        self.to_out = nn.Linear(channels, channels)

    def __call__(self, x, context=None):
        batch, length, channels = x.shape
        if self.cross:
            q = self.to_q(x).reshape(batch, length, self.num_heads, -1)
            kv = self.to_kv(context).reshape(batch, context.shape[1], 2, self.num_heads, -1)
            k, v = kv[:, :, 0], kv[:, :, 1]
        else:
            qkv = self.to_qkv(x).reshape(batch, length, 3, self.num_heads, -1)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = self.q_rms_norm(q).transpose(0, 2, 1, 3)
        k = self.k_rms_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5)
        return self.to_out(output.transpose(0, 2, 1, 3).reshape(batch, length, channels))


class RopeMultiHeadAttention(nn.Module):
    """Denoiser self-attention with real rotary pairs and normalized queries/keys."""

    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qkv = nn.Linear(channels, channels * 3)
        self.q_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.k_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.out = nn.Linear(channels, channels)

    def __call__(self, x, rope_emb):
        batch, length, channels = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        q = self.q_norm(apply_rotary_emb(qkv[:, :, 0], rope_emb)).transpose(0, 2, 1, 3)
        k = self.k_norm(apply_rotary_emb(qkv[:, :, 1], rope_emb)).transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5)
        return self.out(output.transpose(0, 2, 1, 3).reshape(batch, length, channels))
