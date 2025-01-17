# Copyright 2024 The CAM Lab at ETH Zurich.
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
from typing import Any

Tensor = torch.Tensor


class MultiHeadDotProductAttention(nn.Module):
    """Mulit Head Dot Product Attention with querry and key normalization"""

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        normalize_qk: bool = False,
        dropout: float = 0.0,
        device: Any | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super(MultiHeadDotProductAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.normalize_qk = normalize_qk
        self.dropout = dropout
        self.device = device
        self.dtype = dtype

        if emb_dim % num_heads != 0:
            raise ValueError(
                "Embedding Dimension must be divisible through the number of heads"
            )
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.emb_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
            device=self.device,
            dtype=self.dtype,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.multihead_attention.in_proj_weight)
        nn.init.xavier_uniform_(self.multihead_attention.out_proj.weight)

    def forward(
        self, query: Tensor, key: Tensor = None, value: Tensor = None
    ) -> Tensor:
        """Required shape for multihead attention is:

        2D case: (bs, width*height, emb_dim)
        3D case: (bs, length, emb_dim)
        where the length is just the height, width or depth. Used for axial self attention
        """

        if key is None and value is None:
            key = value = query

        elif key is None:
            if value is not None:
                raise ValueError("value can not be not None if key is None")
            key = query

        if value is None:
            value = key

        if self.normalize_qk:
            # L2 normalization across the feature dimension
            query = F.normalize(query, p=2, dim=-1)
            key = F.normalize(key, p=2, dim=-1)

        out, _ = self.multihead_attention(query, key, value)

        return out
