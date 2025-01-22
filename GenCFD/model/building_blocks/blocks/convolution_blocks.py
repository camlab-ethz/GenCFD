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

from typing import Callable, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from GenCFD.utils.model_utils import default_init
from GenCFD.model.building_blocks.layers.convolutions import ConvLayer
from GenCFD.model.building_blocks.layers.residual import CombineResidualWithSkip
from GenCFD.model.building_blocks.blocks.adaptive_scaling import AdaptiveScale

Tensor = torch.Tensor


class ResConv1x(nn.Module):
    """Single-layer residual network with size-1 conv kernels."""

    def __init__(
        self,
        in_channels: int,
        hidden_layer_size: int,
        out_channels: int,
        kernel_dim: int,
        act_fun: Callable[[Tensor], Tensor] = F.silu,
        dtype: torch.dtype = torch.float32,
        scale: float = 1e-10,
        project_skip: bool = False,
        device: Any | None = None,
    ):
        super(ResConv1x, self).__init__()

        self.in_channels = in_channels
        self.hidden_layer_size = hidden_layer_size
        self.out_channels = out_channels
        self.kernel_dim = kernel_dim
        self.act_fun = act_fun
        self.dtype = dtype
        self.scale = scale
        self.project_skip = project_skip
        self.device = device

        self.kernel_size = self.kernel_dim * (1,)

        if self.kernel_dim == 1:
            # case 1
            self.conv1 = nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.hidden_layer_size,
                kernel_size=self.kernel_size,
                dtype=self.dtype,
                device=self.device,
            )
            self.conv2 = nn.Conv1d(
                in_channels=self.hidden_layer_size,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                dtype=self.dtype,
                device=self.device,
            )

        elif self.kernel_dim == 2:
            # case 2
            self.conv1 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.hidden_layer_size,
                kernel_size=self.kernel_size,
                dtype=self.dtype,
                device=self.device,
            )
            self.conv2 = nn.Conv2d(
                in_channels=self.hidden_layer_size,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                dtype=self.dtype,
                device=self.device,
            )

        elif self.kernel_dim == 3:
            # case 3
            self.conv1 = nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=self.hidden_layer_size,
                kernel_size=self.kernel_size,
                device=self.device,
                dtype=self.dtype,
            )
            self.conv2 = nn.Conv3d(
                in_channels=self.hidden_layer_size,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            raise ValueError(
                f"Unsupported input dimension. Expected 1D, 2D or 3D datasets"
            )

        # Initialize weights and biases of the convolution layers
        default_init(self.scale)(self.conv1.weight)
        torch.nn.init.zeros_(self.conv1.bias)
        default_init(self.scale)(self.conv2.weight)
        torch.nn.init.zeros_(self.conv2.bias)

        self.combine_skip = CombineResidualWithSkip(
            residual_channels=self.in_channels,
            skip_channels=self.out_channels,
            project_skip=self.project_skip,
            dtype=self.dtype,
            device=self.device,
        )

    def forward(self, x):

        skip = x.clone()

        x = self.conv1(x)
        x = self.act_fun(x)
        x = self.conv2(x)

        x = self.combine_skip(residual=x, skip=skip)

        return x


class ConvBlock(nn.Module):
    """A basic two-layer convolution block with adaptive scaling in between.

    main conv path:
    --> GroupNorm --> Swish --> Conv -->
        GroupNorm --> FiLM --> Swish --> Dropout --> Conv

    shortcut path:
    --> Linear

    Attributes:
      channels: The number of output channels.
      kernel_sizes: Kernel size for both conv layers.
      padding: The type of convolution padding to use.
      dropout: The rate of dropout applied in between the conv layers.
      film_act_fun: Activation function for the FilM layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        kernel_size: tuple[int, ...],
        padding_mode: str = "circular",
        padding: int = 0,
        stride: int = 1,
        use_bias: bool = True,
        case: int = 2,
        dropout: float = 0.0,
        film_act_fun: Callable[[Tensor], Tensor] = F.silu,
        act_fun: Callable[[Tensor], Tensor] = F.silu,
        dtype: torch.dtype = torch.float32,
        device: Any | None = None,
        **kwargs,
    ):
        super(ConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode
        self.dropout = dropout
        self.film_act_fun = film_act_fun
        self.act_fun = act_fun
        self.dtype = dtype
        self.device = device
        self.padding = padding
        self.stride = stride
        self.use_bias = use_bias
        self.case = case

        self.norm1 = nn.GroupNorm(
            min(max(self.in_channels // 4, 1), 32),
            self.in_channels,
            device=self.device,
            dtype=self.dtype,
        )

        self.conv1 = ConvLayer(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding_mode=self.padding_mode,
            padding=self.padding,
            stride=self.stride,
            use_bias=self.use_bias,
            case=self.case,
            kernel_init=default_init(1.0),
            dtype=self.dtype,
            device=self.device,
            **kwargs,
        )

        self.norm2 = nn.GroupNorm(
            min(max(self.out_channels // 4, 1), 32),
            self.out_channels,
            device=self.device,
            dtype=self.dtype,
        )

        self.film = AdaptiveScale(
            emb_channels=self.emb_channels,
            input_channels=self.out_channels,
            input_dim=self.case,
            act_fun=self.film_act_fun,
            dtype=self.dtype,
            device=self.device,
        )

        self.dropout_layer = nn.Dropout(dropout)

        self.conv2 = ConvLayer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding_mode=self.padding_mode,
            padding=self.padding,
            stride=self.stride,
            use_bias=self.use_bias,
            case=self.case,
            kernel_init=default_init(1.0),
            dtype=self.dtype,
            device=self.device,
        )
        self.res_layer = CombineResidualWithSkip(
            residual_channels=self.out_channels,
            skip_channels=self.in_channels,
            kernel_dim=self.case,
            project_skip=True,
            dtype=self.dtype,
            device=self.device,
        )

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        # ConvBlock per level in the UNet doesn't change it's number of
        # channels or resolution.
        h = x.clone()
        # First block
        h = self.norm1(h)
        h = self.act_fun(h)
        h = self.conv1(h)
        # second block
        h = self.norm2(h)
        h = self.film(h, emb)
        h = self.act_fun(h)
        # For dropout use the following logic: set UNet to .train() or .eval()
        h = self.dropout_layer(h)
        h = self.conv2(h)
        # residual connection
        return self.res_layer(residual=h, skip=x)
