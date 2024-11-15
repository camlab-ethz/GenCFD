# Copyright 2024 The swirl_dynamics Authors.
# Modifications made by the CAM Lab at ETH Zurich.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Sequence

from model.building_blocks.layers.residual import CombineResidualWithSkip
from model.building_blocks.layers.multihead_attention import MultiHeadDotProductAttention
from model.building_blocks.layers.axial_attention import AddAxialPositionEmbedding, AxialSelfAttention
from utils.model_utils import default_init, reshape_jax_torch

Tensor = torch.Tensor

class AttentionBlock(nn.Module):
    """Attention block."""

    def __init__(self, 
                 rng: torch.Generator,
                 num_heads: int = 1,
                 normalize_qk: bool = False, 
                 dtype: torch.dtype = torch.float32,
                 device: Any | None = None):
        super(AttentionBlock, self).__init__()

        self.num_heads = num_heads
        self.normalize_qk = normalize_qk
        self.dtype = dtype
        self.device = device
        self.rng = rng

        self.norm = None
        self.multihead_attention = None

        self.res_layer = CombineResidualWithSkip(
            rng=self.rng,
            dtype=self.dtype, 
            device=self.device
        )
    
    def forward(self, x: Tensor, is_training: bool) -> Tensor:
        # Input x -> (bs, widht*height, c)
        if self.norm is None:
            self.norm = nn.GroupNorm(
                min(max(x.shape[-1] // 4, 1), 32), x.shape[-1],
                device=self.device,
                dtype=self.dtype
            )
        if self.multihead_attention is None:
            self.multihead_attention = MultiHeadDotProductAttention(
                emb_dim=x.shape[-1], 
                num_heads=self.num_heads, 
                rng=self.rng,
                dropout=0.1 if is_training else 0.0,
                device=self.device, 
                dtype=self.dtype
            )

        h = x.clone()
        # GroupNorm requires x -> (bs, c, widht*height)
        h = self.norm(h.permute(0, 2, 1))
        h = h.permute(0, 2, 1) # (bs, width*height, c)
        h = self.multihead_attention(h, h, h)
        h = self.res_layer(residual=h, skip=x)

        return h    


class AxialSelfAttentionBlock(nn.Module):
    """Block consisting of (potentially multiple) axial attention layers."""

    def __init__(
            self,
            rng: torch.Generator,
            attention_axes: int | Sequence[int] = -2,
            add_position_embedding: bool | Sequence[bool] = True,
            num_heads: int | Sequence[int] = 1,
            dtype: torch.dtype = torch.float32,
            device: torch.device = None
        ):
        super(AxialSelfAttentionBlock, self).__init__()
        
        self.rng = rng
        self.dtype = dtype
        self.device = device

        if isinstance(attention_axes, int):
            attention_axes = (attention_axes,)
        self.attention_axes = attention_axes
        num_axes = len(attention_axes)

        if isinstance(add_position_embedding, bool):
            add_position_embedding = (add_position_embedding,) * num_axes
        self.add_position_embedding = add_position_embedding

        if isinstance(num_heads, int):
            num_heads = (num_heads,) * num_axes
        self.num_heads = num_heads

        self.attention_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        self.dense_layers = nn.ModuleList()
        self.pos_emb_layers = nn.ModuleList()
        self.residual_layer = None

        for axis, add_emb, num_head in zip(attention_axes, add_position_embedding, num_heads):
            if add_emb:
                self.pos_emb_layers.append(
                    AddAxialPositionEmbedding(
                        position_axis=axis, 
                        dtype=self.dtype, 
                        device=self.device
                    )
                )
            else:
                self.pos_emb_layers.append(None)
            
            self.norm_layers_1.append(None)
            self.attention_layers.append(None)
            self.norm_layers_2.append(None)
            self.dense_layers.append(None)

    def forward(self, x: Tensor, is_training: bool) -> Tensor:

        # Axial attention ops followed by a projection.
        h = x
        for level, (axis, add_emb, num_head) in enumerate(zip(self.attention_axes, self.add_position_embedding, self.num_heads)):
            if add_emb:
                h = reshape_jax_torch(self.pos_emb_layers[level](reshape_jax_torch(h)))
            
            if self.norm_layers_1[level] is None:
                self.norm_layers_1[level] = nn.GroupNorm(
                    min(max(h.shape[1] // 4, 1), 32),
                    h.shape[1],
                    device=self.device, 
                    dtype=self.dtype
                )
            h = self.norm_layers_1[level](h)

            if self.attention_layers[level] is None:
                self.attention_layers[level] = AxialSelfAttention(
                    num_heads=num_head,
                    rng=self.rng,
                    attention_axis=axis,
                    dropout=0.1 if is_training else 0.0,
                    dtype=self.dtype,
                    device=self.device
                )
            h = reshape_jax_torch(self.attention_layers[level](reshape_jax_torch(h)))

            if self.norm_layers_2[level] is None:
                self.norm_layers_2[level] = nn.GroupNorm(
                    min(max(h.shape[1] // 4, 1), 32),
                    h.shape[1],
                    device=self.device,
                    dtype=self.dtype
                )
            h = self.norm_layers_2[level](h)

            if self.dense_layers[level] is None:
                self.dense_layers[level] = nn.Linear(
                    in_features=h.shape[1],
                    out_features=h.shape[1],
                    device=self.device,
                    dtype=self.dtype
                )
                default_init(1.0)(self.dense_layers[level].weight)
                torch.nn.init.zeros_(self.dense_layers[level].bias)

            h = reshape_jax_torch(self.dense_layers[level](reshape_jax_torch(h)))

        if self.residual_layer is None:
            self.residual_layer = CombineResidualWithSkip(
                rng=self.rng,
                project_skip=h.shape[1] != x.shape[1],
                dtype=self.dtype,
                device=self.device
            )
        h = self.residual_layer(residual=h, skip=x)

        return h


