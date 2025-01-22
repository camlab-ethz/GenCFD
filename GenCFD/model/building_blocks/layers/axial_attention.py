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

"""Axial attention modules."""

import torch.nn as nn
import torch

from GenCFD.model.building_blocks.layers.multihead_attention import (
    MultiHeadDotProductAttention,
)

Tensor = torch.Tensor


class AddAxialPositionEmbedding(nn.Module):
    """Adds trainable axial position embeddings to the inputs."""

    def __init__(
        self,
        position_axis: int,
        spatial_resolution: int,
        input_channels: int,
        initializer: nn.init = nn.init.normal_,
        std: float = 0.02,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ):
        super(AddAxialPositionEmbedding, self).__init__()

        self.initializer = initializer
        self.position_axis = position_axis
        self.spatial_resolution = spatial_resolution
        self.input_channels = input_channels
        self.kernel_dim = len(spatial_resolution)
        self.input_dim = self.kernel_dim + 2  # channel and batch_size in addition
        self.std = std
        self.dtype = dtype
        self.device = device

        pos_axis = self.position_axis
        pos_axis = pos_axis if pos_axis >= 0 else pos_axis + self.input_dim

        if not 0 <= pos_axis < self.input_dim:
            raise ValueError(f"Invalid position ({self.position_axis}) or feature axis")

        self.feat_axis = self.input_dim - 1
        if pos_axis == self.feat_axis:
            raise ValueError(
                f"Position axis ({self.position_axis}) must not coincide with feature"
                f" axis ({self.feat_axis})!"
            )

        unsqueeze_axes = tuple(set(range(self.input_dim)) - {pos_axis, self.feat_axis})
        self.unsqueeze_axes = sorted(unsqueeze_axes)

        self.embedding = nn.Parameter(
            self.initializer(
                torch.empty(
                    (spatial_resolution[pos_axis - 1], input_channels),
                    dtype=self.dtype,
                    device=self.device,
                ),
                std=self.std,
            )
        )

    def forward(self, inputs: Tensor) -> Tensor:
        # Tensor should be off shape: (bs, width, height, depth, c)

        embedding = self.embedding

        if self.unsqueeze_axes:
            for axis in self.unsqueeze_axes:
                embedding = embedding.unsqueeze(dim=axis)

        return inputs + embedding


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_heads=None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ):
        super().__init__()

        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.device = device
        self.dtype = dtype

        self.heads = heads
        self.to_q = nn.Linear(
            dim, dim_hidden, bias=False, device=self.device, dtype=self.dtype
        )
        self.to_kv = nn.Linear(
            dim, 2 * dim_hidden, bias=False, device=self.device, dtype=self.dtype
        )
        self.to_out = nn.Linear(dim_hidden, dim, device=self.device, dtype=self.dtype)

        # self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier uniform distribution."""
        torch.nn.init.xavier_uniform_(self.to_q.weight)
        torch.nn.init.xavier_uniform_(self.to_kv.weight)
        torch.nn.init.xavier_uniform_(self.to_out.weight)

    def forward(self, query, kv=None):
        kv = query if kv is None else kv
        q, k, v = (self.to_q(query), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads
        merge_heads = (
            lambda query: query.reshape(b, -1, h, e)
            .transpose(1, 2)
            .reshape(b * h, -1, e)
        )
        q, k, v = map(merge_heads, (q, k, v))
        dots = torch.einsum("bie,bje->bij", q, k) * (e**-0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum("bij,bje->bie", dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out


class AxialSelfAttention(nn.Module):
    """Axial self-attention for multidimensional inputs."""

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        attention_axis: int = -2,
        dropout: float = 0.0,
        normalize_qk: bool = False,
        use_simple_attention: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ):
        super(AxialSelfAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.attention_axis = attention_axis
        self.dropout = dropout
        self.normalize_qk = normalize_qk
        self.dtype = dtype
        self.device = device

        if use_simple_attention:
            self.attention = SelfAttention(
                dim=self.emb_dim,
                heads=self.num_heads,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            self.attention = MultiHeadDotProductAttention(
                emb_dim=self.emb_dim,
                num_heads=self.num_heads,
                normalize_qk=self.normalize_qk,
                dropout=self.dropout,
                device=self.device,
                dtype=self.dtype,
            )

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

        out = self.attention(query=query)

        out = out.reshape(*inputs.shape)
        out = torch.swapaxes(out, -2, self.attention_axis)

        return out
