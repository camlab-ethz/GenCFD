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
#
# Modifications made by CAM LAB, 09.2024.
# Converted from JAX to PyTorch and made further changes.

"""Residual layer modules."""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Any
from utils.model_utils import reshape_jax_torch

Tensor = torch.Tensor

class CombineResidualWithSkip(nn.Module):
  """Combine residual and skip connections.

  Attributes:
    project_skip: Whether to add a linear projection layer to the skip
      connections. Mandatory if the number of channels are different between
      skip and residual values.
  """
  def __init__(self, 
               rng: torch.Generator,
               project_skip: bool=False, 
               dtype: torch.dtype=torch.float32,
               device: Any | None = None):
    super(CombineResidualWithSkip, self).__init__()

    self.project_skip = project_skip
    self.dtype = dtype
    self.device = device
    self.rng = rng

    self.skip_projection = None

  def forward(self, residual: Tensor, skip: Tensor) -> Tensor:
    # residual, skip (bs, c, w, h, d)
    if self.project_skip:
      if self.skip_projection is None:
        # linear projection layer to match the number of channels
        self.skip_projection = nn.Linear(
          skip.shape[1], 
          residual.shape[1],
          device=self.device,
          dtype=self.dtype
          )
        torch.nn.init.kaiming_uniform_(
          self.skip_projection.weight, a=np.sqrt(5), generator=self.rng
        )
        torch.nn.init.zeros_(self.skip_projection.bias)

      skip = reshape_jax_torch(self.skip_projection(reshape_jax_torch(skip)))
    # combine skip and residual connections
    return (skip + residual) / torch.sqrt(torch.tensor(2.0, dtype=self.dtype, device=self.device))

