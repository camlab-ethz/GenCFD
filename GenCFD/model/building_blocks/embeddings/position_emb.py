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

from typing import Callable, Sequence
import torch
import torch.nn as nn

Tensor = torch.Tensor


def position_embedding(kernel_dim: int, in_shape: Sequence[int], **kwargs) -> nn.Module:
    if kernel_dim == 1:
        return Add1dPosEmbedding(in_shape=in_shape, **kwargs)
    elif kernel_dim == 2:
        return Add2dPosEmbedding(in_shape=in_shape, **kwargs)
    elif kernel_dim == 3:
        return Add3dPosEmbedding(in_shape=in_shape, **kwargs)
    else:
        raise ValueError("Only 1D, 2D, 3D position embeddings are supported.")


class Add1dPosEmbedding(nn.Module):
    """Adds a trainable 1D position embeddings to the inputs."""

    def __init__(
        self,
        in_shape: Sequence[int],
        emb_init: Callable[[Tensor, float, float], None] = nn.init.normal_,
        stddev: float = 0.02,
    ):
        super(Add1dPosEmbedding, self).__init__()

        self.in_shape = token_shape
        self.emb_init = emb_init
        self.stddev = stddev
        self.pos_emb = nn.Parameter(torch.empty(self.in_shape))
        self.emb_init(self.pos_emb, mean=0.0, std=self.stddev)

    def forward(self, x: Tensor) -> Tensor:
        # Input shape of the tensor: (bs, c, l)
        assert len(x.shape) == 3
        return x + self.pos_emb.unsqueeze(0)


class Add2dPosEmbedding(nn.Module):
    """Adds a trainable 2D position embeddings to the inputs."""

    def __init__(
        self,
        in_shape: Sequence[int],
        emb_init: Callable[[Tensor, float, float], None] = nn.init.normal_,
        stddev: float = 0.02,
    ):
        super(Add2dPosEmbedding, self).__init__()

        self.in_shape = in_shape
        self.emb_dim, self.height, self.width = in_shape
        self.emb_init = emb_init
        self.stddev = stddev

        assert self.emb_dim % 2 == 0, "Number of channels must be even"
        self.row_emb = nn.Parameter(torch.empty(self.emb_dim // 2, self.width))
        self.col_emb = nn.Parameter(torch.empty(self.emb_dim // 2, self.height))
        self.emb_init(self.row_emb, mean=0.0, std=self.stddev)
        self.emb_init(self.col_emb, mean=0.0, std=self.stddev)

    def forward(self, x: Tensor) -> Tensor:
        # Input shape of the tensor: (bs, c, h, w)
        assert len(x.shape) == 4

        row_emb = self.row_emb.unsqueeze(1).repeat(1, self.height, 1)  # (c, h, w)
        col_emb = self.col_emb.unsqueeze(-1).repeat(1, 1, self.width)  # (c, h, w)

        pos_emb = torch.cat([col_emb, row_emb], dim=0)

        return x + pos_emb.unsqueeze(0)


class Add3dPosEmbedding(nn.Module):
    """Adds a trainable 2D position embeddings to the inputs."""

    def __init__(
        self,
        input_dim: Sequence[int],
        emb_init: Callable[[Tensor, float, float], None] = nn.init.normal_,
        stddev: float = 0.02,
    ):
        super(Add3dPosEmbedding, self).__init__()

        self.input_dim = input_dim
        self.emb_dim, self.depth, self.height, self.width = input_dim
        self.emb_init = emb_init
        self.stddev = stddev

        assert self.emb_dim % 3 == 0, "Number of channels must be divisible through 3"

        self.row_emb = nn.Parameter(
            torch.empty(self.emb_dim // 3, self.depth, self.width)
        )
        self.col_emb = nn.Parameter(
            torch.empty(self.emb_dim // 3, self.depth, self.height)
        )
        self.depth_emb = nn.Parameter(
            torch.empty(self.emb_dim // 3, self.height, self.width)
        )
        self.emb_init(self.row_emb, mean=0.0, std=self.stddev)
        self.emb_init(self.col_emb, mean=0.0, std=self.stddev)
        self.emb_init(self.depth_emb, mean=0.0, std=self.stddev)

    def forward(self, x: Tensor) -> Tensor:
        # x.shape: (bs, c, depth, height, width)
        assert len(x.shape) == 5

        row_emb = self.row_emb.unsqueeze(2).repeat(1, 1, self.height, 1)
        col_emb = self.col_emb.unsqueeze(-1).repeat(1, 1, 1, self.width)
        depth_emb = self.depth_emb.unsqueeze(1).repeat(1, self.depth, 1, 1)

        pos_emb = torch.cat([depth_emb, col_emb, row_emb], dim=0)

        return x + pos_emb
