# Adapted from speridlabs/idol-sdk idol/models/decoder.py, Apache-2.0.
"""Preliminary FP32 octree probability decoder."""

import mlx.core as mx
import mlx.nn as nn

from .layers import ModulatedTransformerCrossOnlyBlock, PcdAbsolutePositionEmbedderv2, SinusoidalEmbedder


class OctreeProbabilityFixedlenDecoder(nn.Module):
    """Predict eight child logits per cell; sampling is a separate operation."""

    def __init__(self, model_channels, cond_channels, num_blocks, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.in_proj = nn.Linear(3, model_channels)
        self.pos_embedder = PcdAbsolutePositionEmbedderv2(channels=model_channels, in_channels=3)
        self.input_layer = nn.Linear(model_channels, model_channels)
        self.l_embedder = SinusoidalEmbedder(model_channels, max_period=1024, two_pi=True)
        self.adaLN_modulation = nn.Sequential(nn.silu, nn.Linear(model_channels, 6 * model_channels))
        self.blocks = [ModulatedTransformerCrossOnlyBlock(model_channels, cond_channels, num_heads, mlp_ratio)
                       for _ in range(num_blocks)]
        self.out_proj = nn.Linear(model_channels, 8)

    def __call__(self, x, level, cond):
        batch, length, _ = x.shape
        positions = self.pos_embedder(x.reshape(-1, 3)).reshape(batch, length, -1)
        h = self.input_layer(self.in_proj(x) + positions)
        modulation = self.adaLN_modulation(self.l_embedder(level.astype(h.dtype)))
        for block in self.blocks:
            h = block(h, modulation, cond)
        return self.out_proj(mx.fast.layer_norm(h, None, None, 1e-5))
