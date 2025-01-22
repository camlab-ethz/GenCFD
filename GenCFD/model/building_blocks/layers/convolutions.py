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

"""Convolution layers."""

from typing import Literal, Sequence, Any, Callable
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import functools

Tensor = torch.Tensor


def ConvLayer(
    in_channels: int,
    out_channels: int,
    kernel_size: int | Sequence[int],
    padding_mode: str,
    padding: int = 0,
    stride: int = 1,
    use_bias: bool = True,
    use_local: bool = False,
    case: int = 2,
    kernel_init: Callable = None,
    bias_init: Callable = torch.nn.init.zeros_,
    dtype: torch.dtype = torch.float32,
    device: Any | None = None,
    **kwargs,
) -> nn.Module:
    """Factory for different types of convolution layers.

    Where the last part requires a case differentiation:
    case == 1: 1D (bs, c, width)
    case == 2: 2D (bs, c, height, width)
    case == 3: 3D (bs, c, depth, height, width)
    """
    if isinstance(padding_mode, str) and padding_mode.lower() in ["lonlat", "latlon"]:
        if not (isinstance(kernel_size, tuple) and len(kernel_size) == 2):
            raise ValueError(
                f"kernel size {kernel_size} must be a length-2 tuple "
                f"for convolution type {padding_mode}."
            )
        return LatLonConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            order=padding_mode.lower(),
            dtype=dtype,
            device=device,
            **kwargs,
        )

    elif use_local:
        return ConvLocal2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            use_bias=use_bias,
            device=device,
            dtype=dtype,
        )
    else:
        if case == 1:
            conv_layer = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding_mode=padding_mode.lower(),
                padding=padding,
                stride=stride,
                bias=use_bias,
                device=device,
                dtype=dtype,
            )
        elif case == 2:
            conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding_mode=padding_mode.lower(),
                padding=padding,
                stride=stride,
                bias=use_bias,
                device=device,
                dtype=dtype,
            )
        elif case == 3:
            conv_layer = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding_mode=padding_mode.lower(),
                padding=padding,
                stride=stride,
                bias=use_bias,
                device=device,
                dtype=dtype,
            )

    # Initialize weights and biases
    if kernel_init is not None:
        kernel_init(conv_layer.weight)
        bias_init(conv_layer.bias)

    return conv_layer


class ConvLocal2d(nn.Module):
    """Customized locally connected 2D convolution (ConvLocal) for PyTorch"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride: int = 1,
        padding: int = 0,
        padding_mode: str = "constant",
        use_bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        super(ConvLocal2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.padding_mode = padding_mode
        self.use_bias = use_bias
        self.device = device
        self.dtype = dtype

        # Weights for each spatial location (out_height x out_width)
        self.weights = None

        if self.use_bias:
            self.bias = nn.Parameter(
                torch.zeros(out_channels, dtype=self.dtype, device=self.device)
            )
        else:
            self.bias = None

    def forward(self, x):
        if len(x.shape) < 4:
            raise ValueError(
                f"Local 2D Convolution with shape length of 4 instead of {len(x.shape)}"
            )

        # Input dim: (batch_size, in_channels, height, width)
        # width, height = lat, lon
        batch_size, in_channels, height, width = x.shape

        if self.padding > 0:
            x = F.pad(
                x,
                [self.padding, self.padding, self.padding, self.padding],
                mode=self.padding_mode,
                value=0,
            )

        out_height = (height - self.kernel_size[0] + 2 * self.padding) // self.stride[
            0
        ] + 1
        out_width = (width - self.kernel_size[1] + 2 * self.padding) // self.stride[
            1
        ] + 1

        # Initialize weights
        if self.weights is None:
            self.weights = nn.Parameter(
                torch.empty(
                    out_height,
                    out_width,
                    self.out_channels,
                    in_channels,
                    self.kernel_size[0],
                    self.kernel_size[1],
                    device=self.device,  # x.device
                    dtype=self.dtype,
                )
            )
            torch.nn.init.xavier_uniform_(self.weights)

        output = torch.zeros(
            (batch_size, self.out_channels, out_height, out_width),
            dtype=self.dtype,
            device=self.device,
        )

        # manually scripted convolution.
        for i in range(out_height):
            for j in range(out_width):
                patch = x[
                    :,
                    :,
                    i * self.stride[0] : i * self.stride[0] + self.kernel_size[0],
                    j * self.stride[1] : j * self.stride[1] + self.kernel_size[1],
                ]
                # Sums of the product based on Einstein's summation convention
                output[:, :, i, j] = torch.einsum(
                    "bchw, ocwh->bo", patch, self.weights[i, j]
                )

        if self.use_bias:
            bias_shape = [1] * len(x.shape)
            bias_shape[1] = -1
            output += self.bias.view(bias_shape)

        return output


class LatLonConv(nn.Module):
    """2D convolutional layer adapted to inputs a lot-lon grid"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        order: Literal["latlon", "lonlat"] = "latlon",
        use_bias: bool = True,
        strides: tuple[int, int] = (1, 1),
        use_local: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Any | None = None,
        **kwargs,
    ):
        super(LatLonConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.order = order
        self.use_bias = use_bias
        self.strides = strides
        self.use_local = use_local
        self.dtype = dtype
        self.device = device

        if self.use_local:
            self.conv = ConvLocal2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=strides,
                bias=use_bias,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=strides,
                bias=use_bias,
                padding=0,
                device=self.device,
                dtype=self.dtype,
            )

    def forward(self, inputs):
        """Applies lat-lon and lon-lat convolution with edge and circular padding"""
        if len(inputs.shape) < 4:
            raise ValueError(f"Input must be 4D or higher: {inputs.shape}.")

        if self.kernel_size[0] % 2 == 0 or self.kernel_size[1] % 2 == 0:
            raise ValueError(f"Current kernel size {self.kernel_size} must be odd.")

        if self.order == "latlon":
            lon_axis, lat_axis = (-3, -2)
            lat_pad, lon_pad = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        elif self.order == "lonlat":
            lon_axis, lat_axis = (-3, -2)
            lon_pad, lat_pad = self.kernel_size[1] // 2, self.kernel_size[0] // 2
            # TODO: There is no difference lon_axis and lat_axis in "lonlat" should be switched?
        else:
            raise ValueError(
                f"Unrecogniized order {self.order} - 'loatlon' or 'lonlat expected."
            )

        # Circular padding to longitudinal (lon) axis
        padded_inputs = F.pad(inputs, [0, 0, lon_pad, lon_pad], mode="circular")
        # Edge padding to latitudinal (lat) axis
        padded_inputs = F.pad(padded_inputs, [lat_pad, lat_pad, 0, 0], mode="replicate")

        return self.conv(padded_inputs)


class DownsampleConv(nn.Module):
    """Downsampling layer through strided convolution.

    Arguments:
      in_channels: number of input channels
      out_channels: number of output channels
      ratios: downsampling ratio for the resolution, increase of the channel dimension
      case: dimensionality of the dataset, 1D, 2D and 3D (int: 1, 2, or 3)
      use_bias:  If True, adds a learnable bias to the output. Default: True
      kernel_init: initializations for the convolution weights
      bias_init: initializtations for the convolution bias values
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_resolution: Sequence[int],
        ratios: Sequence[int],
        case: int,
        use_bias: bool = True,
        kernel_init: Callable = torch.nn.init.kaiming_uniform_,
        bias_init: Callable = torch.nn.init.zeros_,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        super(DownsampleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_dim = len(spatial_resolution)  # Spatial dimension of dataset
        self.ratios = ratios
        self.bias_init = bias_init
        self.dtype = dtype
        self.device = device

        dataset_shape = self.kernel_dim + 2  # spatial resolution + channel + batch_size

        # Check if input has the correct shape and size to be downsampled!
        if dataset_shape <= len(self.ratios):
            raise ValueError(
                f"Inputs ({dataset_shape}) for downsampling must have at least 1 more dimension "
                f"than that of 'ratios' ({self.ratios})."
            )

        if not all(s % r == 0 for s, r in zip(spatial_resolution, self.ratios)):
            raise ValueError(
                f"Input dimensions (spatial) {spatial_resolution} must divide the "
                f"downsampling ratio {self.ratios}."
            )

        self.use_bias = use_bias
        if kernel_init is torch.nn.init.kaiming_uniform_:
            self.kernel_init = functools.partial(kernel_init, a=np.sqrt(5))
        else:
            self.kernel_init = kernel_init

        # For downsampling padding = 0 and stride > 1
        if case == 1:
            self.conv_layer = nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=ratios,
                stride=ratios,
                bias=use_bias,
                padding=0,
                device=self.device,
                dtype=self.dtype,
            )

        elif case == 2:
            self.conv_layer = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=ratios,
                stride=ratios,
                bias=use_bias,
                padding=0,
                device=self.device,
                dtype=self.dtype,
            )

        elif case == 3:
            self.conv_layer = nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=ratios,
                stride=ratios,
                bias=use_bias,
                padding=0,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            raise ValueError(f"Dataset dimension should either be 1D, 2D or 3D")

        # Initialize with variance_scaling
        # Only use this if the activation function is ReLU or smth. similar
        self.kernel_init(self.conv_layer.weight)
        self.bias_init(self.conv_layer.bias)

    def forward(self, inputs):
        """Applies strided convolution for downsampling."""

        return self.conv_layer(inputs)
