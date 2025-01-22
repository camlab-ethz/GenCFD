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

"""Residual layer modules."""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Any
from GenCFD.utils.model_utils import reshape_jax_torch

Tensor = torch.Tensor


class CombineResidualWithSkip(nn.Module):
    """Combine residual and skip connections.

    Attributes:
      project_skip: Whether to add a linear projection layer to the skip
        connections. Mandatory if the number of channels are different between
        skip and residual values.
    """

    def __init__(
        self,
        residual_channels: int,
        skip_channels: int,
        kernel_dim: int = None,
        project_skip: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Any | None = None,
    ):
        super(CombineResidualWithSkip, self).__init__()

        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.kernel_dim = kernel_dim
        self.project_skip = project_skip
        self.dtype = dtype
        self.device = device

        if residual_channels != skip_channels and not project_skip:
            raise ValueError(
                f"Residual tensor has {residual_channels}, Skip tensor has {skip_channels}. "
                f"Set project_skip to True to resolve this mismatch."
            )

        if self.residual_channels and self.skip_channels and self.project_skip:
            self.skip_projection = nn.Linear(
                skip_channels, residual_channels, device=self.device, dtype=self.dtype
            )
            torch.nn.init.kaiming_uniform_(self.skip_projection.weight, a=np.sqrt(5))
            torch.nn.init.zeros_(self.skip_projection.bias)
        else:
            self.skip_projection = None

    def forward(self, residual: Tensor, skip: Tensor) -> Tensor:
        # residual, skip (bs, c, w, h, d)
        if self.project_skip:
            skip = reshape_jax_torch(
                self.skip_projection(reshape_jax_torch(skip, self.kernel_dim)),
                self.kernel_dim,
            )

        return (skip + residual) / math.sqrt(2)
