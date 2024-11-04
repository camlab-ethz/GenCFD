# Copyright 2024 The swirl_dynamics Authors.
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

"""Axial attention modules."""

import torch.nn as nn
import torch

from model.building_blocks.layers.multihead_attention import MultiHeadDotProductAttention


Tensor = torch.Tensor


class AddAxialPositionEmbedding(nn.Module):
    """Adds trainable axial position embeddings to the inputs."""

    def __init__(self, 
               position_axis: int, 
               initializer: nn.init = nn.init.normal_,
               std: float = 0.02,
               dtype: torch.dtype = torch.float32,
               device: torch.device = None
        ):
        super(AddAxialPositionEmbedding, self).__init__()

        self.initializer = initializer
        self.position_axis = position_axis
        self.std = std
        self.dtype = dtype
        self.device = device

        self.embedding = None


    def forward(self, inputs: Tensor) -> Tensor:
        # Tensor should be off shape: (bs, width, height, c)
        pos_axis = self.position_axis
        pos_axis = pos_axis if pos_axis >= 0 else pos_axis + inputs.ndim

        if not 0 <= pos_axis < inputs.ndim:
            raise ValueError(
                f"Invalid position ({self.position_axis}) or feature axis"
            )

        feat_axis = inputs.ndim - 1
        if pos_axis == feat_axis:
            raise ValueError(
                f"Position axis ({self.position_axis}) must not coincide with feature"
                f" axis ({feat_axis})!"
            )

        unsqueeze_axes = tuple(set(range(inputs.ndim)) - {pos_axis, feat_axis})

        if self.embedding is None:
            self.embedding = nn.Parameter(
                self.initializer(
                    torch.empty(
                        (inputs.shape[pos_axis], inputs.shape[feat_axis]), 
                        dtype=self.dtype, 
                        device=self.device),
                    std=self.std
                )
            )
        
        embedding = self.embedding
        
        if unsqueeze_axes:
            for axis in sorted(unsqueeze_axes):
                embedding = embedding.unsqueeze(dim=axis)

        return inputs + embedding


class AxialSelfAttention(nn.Module):
    """Axial self-attention for multidimensional inputs."""

    def __init__(self,
                 num_heads: int,
                 rng: torch.Generator,
                 attention_axis: int = -2,
                 dropout: float = 0.0,
                 normalize_qk: bool = False,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = None
        ):
        super(AxialSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_axis = attention_axis
        self.dropout = dropout
        self.normalize_qk = normalize_qk
        self.dtype = dtype 
        self.rng = rng
        self.device = device

        self.attention = None


    def forward(self, inputs: Tensor) -> Tensor:
        """Applies axial self-attention to the inputs.
        
        inputs: Tensor should have the shape (bs, width, height, depth, c)
            where c here is the embedding dimension
        """

        if self.attention_axis == -1 or self.attention_axis == inputs.ndim - 1:
            raise ValueError(
                f"Attention axis ({self.attention_axis}) cannot be the last axis,"
                " which is treated as the features!"
            )

        inputs = torch.swapaxes(inputs, self.attention_axis, -2)
        query = inputs.reshape(-1, *inputs.shape[-2:])

        if self.attention is None:
            self.attention = MultiHeadDotProductAttention(
                emb_dim=inputs.shape[-1],
                num_heads=self.num_heads,
                rng=self.rng,
                normalize_qk=self.normalize_qk,
                dropout=self.dropout,
                device=self.device,
                dtype=self.dtype
            )

        out = self.attention(query=query)

        out = out.reshape(*inputs.shape)
        out = torch.swapaxes(out, -2, self.attention_axis)

        return out
