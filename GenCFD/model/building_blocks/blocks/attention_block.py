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

from GenCFD.model.building_blocks.layers.residual import CombineResidualWithSkip
from GenCFD.model.building_blocks.layers.multihead_attention import (
    MultiHeadDotProductAttention,
)
from GenCFD.model.building_blocks.layers.axial_attention import (
    AddAxialPositionEmbedding,
    AxialSelfAttention,
)
from GenCFD.utils.model_utils import default_init, reshape_jax_torch

Tensor = torch.Tensor


class AttentionBlock(nn.Module):
    """Attention block."""

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 1,
        normalize_qk: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Any | None = None,
    ):
        super(AttentionBlock, self).__init__()

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.normalize_qk = normalize_qk
        self.dtype = dtype
        self.device = device

        self.norm = nn.GroupNorm(
            min(max(self.in_channels // 4, 1), 32),
            self.in_channels,
            device=self.device,
            dtype=self.dtype,
        )

        self.multihead_attention = MultiHeadDotProductAttention(
            emb_dim=self.in_channels,
            num_heads=self.num_heads,
            dropout=0.1,
            device=self.device,
            dtype=self.dtype,
        )

        self.res_layer = CombineResidualWithSkip(
            residual_channels=in_channels,
            skip_channels=in_channels,
            dtype=self.dtype,
            device=self.device,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Input x -> (bs, widht*height, c)
        h = x.clone()
        # GroupNorm requires x -> (bs, c, widht*height)
        h = self.norm(h.permute(0, 2, 1))
        h = h.permute(0, 2, 1)  # (bs, width*height, c)
        h = self.multihead_attention(h, h, h)  # Selfattention
        h = self.res_layer(residual=h, skip=x)

        return h


class AxialSelfAttentionBlock(nn.Module):
    """Block consisting of (potentially multiple) axial attention layers."""

    def __init__(
        self,
        in_channels: int,
        spatial_resolution: Sequence[int],
        attention_axes: int | Sequence[int] = -2,
        add_position_embedding: bool | Sequence[bool] = True,
        num_heads: int | Sequence[int] = 1,
        normalize_qk: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ):
        super(AxialSelfAttentionBlock, self).__init__()

        self.in_channels = in_channels
        self.dtype = dtype
        self.device = device
        self.kernel_dim = len(spatial_resolution)
        self.normalize_qk = normalize_qk

        # permute spatial resolution since the following transformation in the 3D case is being done:
        # (bs, c, w, h, d) -> (bs, d, h, w, c) thus the resolution changes (w, h, d) -> (d, h, w)
        spatial_resolution = spatial_resolution[::-1]

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

        for level, (axis, add_emb, num_head) in enumerate(
            zip(self.attention_axes, self.add_position_embedding, self.num_heads)
        ):
            if add_emb:
                self.pos_emb_layers.append(
                    AddAxialPositionEmbedding(
                        position_axis=axis,
                        spatial_resolution=spatial_resolution,
                        input_channels=self.in_channels,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )

            self.norm_layers_1.append(
                nn.GroupNorm(
                    min(max(self.in_channels // 4, 1), 32),
                    self.in_channels,
                    device=self.device,
                    dtype=self.dtype,
                )
            )

            self.attention_layers.append(
                AxialSelfAttention(
                    emb_dim=self.in_channels,
                    num_heads=num_head,
                    attention_axis=axis,
                    dropout=0.1,
                    normalize_qk=self.normalize_qk,
                    dtype=self.dtype,
                    device=self.device,
                )
            )

            self.norm_layers_2.append(
                nn.GroupNorm(
                    min(max(self.in_channels // 4, 1), 32),
                    self.in_channels,
                    device=self.device,
                    dtype=self.dtype,
                )
            )

            self.dense_layers.append(
                nn.Linear(
                    in_features=in_channels,
                    out_features=in_channels,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            default_init(1.0)(self.dense_layers[level].weight)
            torch.nn.init.zeros_(self.dense_layers[level].bias)

        self.residual_layer = CombineResidualWithSkip(
            residual_channels=in_channels,
            skip_channels=in_channels,
            kernel_dim=self.kernel_dim,
            project_skip=False,
            dtype=self.dtype,
            device=self.device,
        )

    def forward(self, x: Tensor) -> Tensor:

        # Axial attention ops followed by a projection.
        h = x.clone()
        for level, (axis, add_emb, num_head) in enumerate(
            zip(self.attention_axes, self.add_position_embedding, self.num_heads)
        ):
            if add_emb:
                # Embedding
                h = reshape_jax_torch(
                    self.pos_emb_layers[level](reshape_jax_torch(h, self.kernel_dim)),
                    self.kernel_dim,
                )
            # Group Normalization
            h = self.norm_layers_1[level](h)
            # Attention Layer
            h = reshape_jax_torch(
                self.attention_layers[level](reshape_jax_torch(h, self.kernel_dim)),
                self.kernel_dim,
            )
            # Group Normalization
            h = self.norm_layers_2[level](h)
            # Dense Layer
            h = reshape_jax_torch(
                self.dense_layers[level](reshape_jax_torch(h, self.kernel_dim)),
                self.kernel_dim,
            )
        # Residual Connection out of the Loop!
        h = self.residual_layer(residual=h, skip=x)

        return h
