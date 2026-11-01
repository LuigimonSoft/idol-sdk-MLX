# Adapted from speridlabs/idol-sdk idol/models/decoder.py, Apache-2.0.
"""Preliminary FP32 Gaussian decoder feature network."""

import mlx.core as mx
import mlx.nn as nn

from .layers import PcdAbsolutePositionEmbedderv2, TransformerCrossBlock


class GaussianFixedlenDecoder(nn.Module):
    """Predict raw per-anchor Gaussian features; attribute conversion is separate."""

    def __init__(self, in_channels, model_channels, cond_channels, num_blocks, num_heads,
                 representation_config, mlp_ratio=4.0):
        super().__init__()
        if in_channels != 3:
            raise ValueError("in_channels must be 3 for 3D anchor positions.")
        if not isinstance(representation_config, dict):
            raise ValueError("representation_config must be a dictionary.")
        per_anchor = representation_config.get("num_gaussians")
        if isinstance(per_anchor, bool) or not isinstance(per_anchor, int) or per_anchor < 1:
            raise ValueError("num_gaussians must be a positive integer.")
        self.in_proj = nn.Linear(in_channels, model_channels)
        self.pos_embedder = PcdAbsolutePositionEmbedderv2(channels=model_channels, in_channels=3)
        self.input_layer = nn.Linear(model_channels, model_channels)
        self.blocks = [TransformerCrossBlock(model_channels, cond_channels, num_heads, mlp_ratio)
                       for _ in range(num_blocks)]
        self.out_proj = nn.Linear(model_channels, per_anchor * 15)
        # The upstream checkpoint includes these buffers for attribute conversion.
        self.points_offset_perturbation = mx.zeros((per_anchor, 3), dtype=mx.float32)
        self.base_offset_scale = mx.zeros((), dtype=mx.float32)

    def __call__(self, points, cond):
        batch, length, channels = points.shape
        pos = self.pos_embedder(points.reshape(-1, channels)).reshape(batch, length, -1)
        out = self.input_layer(self.in_proj(points) + pos)
        for block in self.blocks:
            out = block(out, cond)
        return self.out_proj(mx.fast.layer_norm(out, None, None, 1e-5))
