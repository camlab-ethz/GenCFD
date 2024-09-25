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
from typing import Callable

Tensor = torch.Tensor

class FourierEmbedding(nn.Module):
    """Fourier embedding."""

    def __init__(self, dims: int=64, max_freq: float=2e4,
                 projection: bool=True, 
                 act_fun: Callable[[Tensor], Tensor]=F.silu,
                 dtype: torch.dtype=torch.float32):
        super(FourierEmbedding, self).__init__()

        self.dims = dims
        self.max_freq = max_freq
        self.projection = projection
        self.act_fun = act_fun
        self.dtype = dtype

        self.lin_layer1 = None
        self.lin_layer2 = None

    
    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 1, "Input tensor must be 1D"
        logfreqs = torch.linspace(0, torch.log(torch.tensor(self.max_freq)), 
                                  self.dims // 2)
        
        x_proj = torch.pi * torch.exp(logfreqs)[None, :] * x[:, None]
        x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        # now x_proj is a 2D tensor

        if self.projection:
            if self.lin_layer1 is None or self.lin_layer2 is None:
                self.lin_layer1 = nn.Linear(x_proj.shape[1], 2 * self.dims)
                self.lin_layer2 = nn.Linear(2* self.dims, self.dims)
            
            x_proj = self.lin_layer1(x_proj)
            x_proj = self.act_fun(x_proj)
            x_proj = self.lin_layer2(x_proj)

        return x_proj
        
