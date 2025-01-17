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
from typing import Callable, Any

Tensor = torch.Tensor


class FourierEmbedding(nn.Module):
    """Fourier embedding."""

    def __init__(
        self,
        dims: int = 64,
        max_freq: float = 2e4,
        projection: bool = True,
        act_fun: Callable[[Tensor], Tensor] = F.silu,
        dtype: torch.dtype = torch.float32,
        device: Any | None = None,
        max_val: float = 1e6,  # for numerical stability
    ):
        super(FourierEmbedding, self).__init__()

        self.dims = dims
        self.max_freq = max_freq
        self.projection = projection
        self.act_fun = act_fun
        self.dtype = dtype
        self.device = device
        self.max_val = max_val

        logfreqs = torch.linspace(
            0,
            torch.log(
                torch.tensor(self.max_freq, dtype=self.dtype, device=self.device)
            ),
            self.dims // 2,
            dtype=self.dtype,
            device=self.device,
        )

        # freqs are constant and scaled with pi!
        const_freqs = torch.pi * torch.exp(logfreqs)[None, :]  # Shape: (1, dims//2)

        # Store freqs as a non-trainable buffer also to ensure device and dtype transfers
        self.register_buffer("const_freqs", const_freqs)

        if self.projection:
            self.lin_layer1 = nn.Linear(
                self.dims, 2 * self.dims, dtype=self.dtype, device=self.device
            )
            self.lin_layer2 = nn.Linear(
                2 * self.dims, self.dims, dtype=self.dtype, device=self.device
            )

    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 1, "Input tensor must be 1D"

        # Use the registered buffer const_freqs
        x_proj = self.const_freqs * x[:, None]
        # x_proj is now a 2D tensor
        x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        # clamping values to avoid running into numerical instability!
        x_proj = torch.clamp(x_proj, min=-self.max_val, max=self.max_val)

        if self.projection:
            x_proj = self.lin_layer1(x_proj)
            x_proj = self.act_fun(x_proj)
            x_proj = self.lin_layer2(x_proj)

        return x_proj
