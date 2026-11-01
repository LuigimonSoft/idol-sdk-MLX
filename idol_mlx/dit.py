# Adapted from speridlabs/idol-sdk idol/models/dit.py, itself adapted from
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_wan.py
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Preliminary FP32 MLX denoiser architecture."""

import mlx.core as mx
import mlx.nn as nn

from .layers import (MLP, PcdAbsolutePositionEmbedder, RePo3DRotaryEmbedding,
                     SinusoidalEmbedder, UnifiedTransformerBlock)


class LatentSeqMMFlowModel(nn.Module):
    """Denoiser forward path for FP32 arrays; trained-weight loading is separate."""

    def __init__(self, q_token_length, in_channels, cam_channels, model_channels,
                 cond_channels, cond2_channels, num_blocks, num_refiner_blocks=2,
                 num_heads=16, num_head_channels=64, mlp_ratio=4.0):
        super().__init__()
        if model_channels != num_heads * num_head_channels:
            raise ValueError("model_channels must equal num_heads * num_head_channels.")
        self.q_token_length = q_token_length
        self.in_channels = in_channels
        self.cam_channels = cam_channels

        def blocks(count, modulation=True):
            return [UnifiedTransformerBlock(model_channels, num_heads, mlp_ratio, modulation)
                    for _ in range(count)]

        def rope_layers(count):
            return [RePo3DRotaryEmbedding(model_channels, num_heads, num_head_channels)
                    for _ in range(count)]

        self.t_embedder = SinusoidalEmbedder(model_channels)
        self.adaLN_modulation = nn.Sequential(nn.silu, nn.Linear(model_channels, 6 * model_channels))
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.cond_embedder = nn.Linear(cond_channels, model_channels)
        self.cond_embedder2 = nn.Linear(cond2_channels, model_channels)
        self.pos_pe = mx.zeros((1, q_token_length, 3))
        self.pos_embedder = PcdAbsolutePositionEmbedder(model_channels)
        self.noise_repo_layers = rope_layers(num_refiner_blocks)
        self.context_repo_layers = rope_layers(num_refiner_blocks)
        self.repo_layers = rope_layers(num_blocks)
        self.noise_refiner = blocks(num_refiner_blocks)
        self.context_refiner = blocks(num_refiner_blocks, modulation=False)
        self.cam_refiner = MLP(cam_channels, model_channels, model_channels, mlp_layer_num=num_refiner_blocks)
        self.blocks = blocks(num_blocks)
        self.shift_table = mx.zeros((1, 2, model_channels))
        self.out_layer = nn.Linear(model_channels, in_channels)
        self.cam_out_layer = nn.Linear(model_channels, cam_channels)

    def __call__(self, x, t, cond):
        z, cam = x["latent"], x["camera"]
        h_x = self.input_layer(z)
        h_cond = self.cond_embedder(cond["feature1"]) + self.cond_embedder2(cond["feature2"])
        t_emb = self.t_embedder(t)
        t_mod = self.adaLN_modulation(t_emb)
        h_x = h_x + self.pos_embedder(self.pos_pe)
        for repo, block in zip(self.noise_repo_layers, self.noise_refiner):
            h_x = block(h_x, t_mod, repo(h_x))
        for repo, block in zip(self.context_repo_layers, self.context_refiner):
            h_cond = block(h_cond, None, repo(h_cond))
        h_cam = self.cam_refiner(cam)
        h = mx.concatenate([h_x, h_cond, h_cam], axis=1)
        for repo, block in zip(self.repo_layers, self.blocks):
            h = block(h, t_mod, repo(h))
        h_x = mx.fast.layer_norm(h[:, :z.shape[1]].astype(z.dtype), None, None, 1e-5)
        h_cam = mx.fast.layer_norm(h[:, -cam.shape[1]:].astype(cam.dtype), None, None, 1e-5)
        shift, scale = mx.split(self.shift_table + t_emb[:, None, :], 2, axis=1)
        h_x = h_x * (1 + scale) + shift
        h_cam = h_cam * (1 + scale) + shift
        return {"latent": self.out_layer(h_x), "camera": self.cam_out_layer(h_cam)}
