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

"""Shifts and reshapes channels to the spatial dimensions

Equivalent for the 2D case to nn.PixelShuffle
"""

import torch
import math
import torch.nn as nn
import numpy as np
from typing import Sequence
from GenCFD.model.building_blocks.layers.convolutions import ConvLayer

from GenCFD.utils.model_utils import reshape_jax_torch

Tensor = torch.Tensor


class ChannelToSpace(nn.Module):
    """Reshapes data from the channel to spatial dims as a way to upsample.

    As an example, for an input of shape (*batch, x, y, z) and block_shape of
    (a, b), additional spatial dimensions are first formed from the channel
    dimension (always the last one), i.e. reshaped into
    (*batch, x, y, a, b, z//(a*b)). Then the new axes are interleaved with the
    original ones to arrive at shape (*batch, x, a, y, b, z//(a*b)). Finally, the
    new axes are merged with the original axes to yield final shape
    (*batch, x*a, y*b, z//(a*b)).

    Args:
      inputs: The input array to upsample.
      block_shape: The shape of the block that will be formed from the channel
        dimension. The number of elements (i.e. prod(block_shape) must divide the
        number of channels).
      kernel_dim: Defines the dimension of the input 1D, 2D or 3D
      spatial_resolution: Tuple with the spatial resolution components

    Returns:
      The upsampled array.
    """

    def __init__(
        self,
        block_shape: Sequence[int],
        in_channels: int,
        kernel_dim: int,
        spatial_resolution: Sequence[int],
    ):
        super(ChannelToSpace, self).__init__()

        self.block_shape = block_shape
        self.in_channels = in_channels
        self.kernel_dim = kernel_dim
        # Since also here a transformation happens from (bs, c, w, h, d) -> (bs, d, h, w, c)
        # Thus for the spatial_resolution: (w, h, d) -> (d, h, w)
        spatial_resolution = spatial_resolution[::-1]
        self.spatial_resolution = spatial_resolution
        self.input_dim = kernel_dim + 2  # batch size and channel dimensions are added

        if not self.input_dim > len(self.block_shape):
            raise ValueError(
                f"Ndim of `x` ({self.input_dim}) expected to be higher than the length of"
                f" `block_shape` {len(self.block_shape)}."
            )

        if self.in_channels % math.prod(self.block_shape) != 0:
            raise ValueError(
                f"The number of channels in the input ({self.in_channels}) must be"
                f" divisible by the block size ({math.prod(self.block_shape)})."
            )

        new_spatial_resolution = [
            self.spatial_resolution[i] * self.block_shape[i]
            for i in range(len(self.spatial_resolution))
        ]
        new_spatial_resolution = tuple(new_spatial_resolution)
        self.out_channels = self.in_channels // math.prod(self.block_shape)
        self.new_shape = (-1,) + new_spatial_resolution + (self.out_channels,)

        # Further precomputation
        batch_ndim = self.input_dim - len(self.block_shape) - 1
        # Interleave old and new spatial axes.
        spatial_axes = [i for i in range(1, 2 * len(self.block_shape) + 1)]
        reshaped = [
            spatial_axes[i : i + len(self.block_shape)]
            for i in range(0, len(spatial_axes), len(self.block_shape))
        ]
        permuted = list(map(list, zip(*reshaped)))
        # flattened and spatial_axes is reshaped to column major row
        self.new_axes = tuple([item for sublist in permuted for item in sublist])

        # compute permutation axes:
        self.permutation_axes = (
            tuple(range(batch_ndim))
            + self.new_axes
            + (len(self.new_axes) + batch_ndim,)
        )

    def forward(self, inputs: Tensor) -> Tensor:

        inputs = reshape_jax_torch(inputs, self.kernel_dim)
        x = torch.reshape(
            inputs,
            (-1,)
            + self.spatial_resolution
            + tuple(self.block_shape)
            + (self.out_channels,),
        )
        x = x.permute(self.permutation_axes)
        reshaped_tensor = torch.reshape(x, self.new_shape)

        return reshape_jax_torch(reshaped_tensor, self.kernel_dim)


class LearnablePixelShuffle3D(nn.Module):
    """
    Learnable 3D Pixel Shuffle: Combines deterministic channel-to-space rearrangement
    with a learnable convolution for added flexibility.
    """

    def __init__(
        self,
        in_channels: int,
        upscale_factor: int,
        kernel_dim: int,
        padding_method: str,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ):
        """
        Args:
            in_channels: Number of input channels.
            upscale_factor: Factor by which to upscale spatial dimensions.
        """
        super(LearnablePixelShuffle3D, self).__init__()

        # Check divisibility
        block_size = upscale_factor**3
        if in_channels % block_size != 0:
            raise ValueError(
                f"Input channels ({in_channels}) must be divisible by "
                f"block_size ({block_size})."
            )

        self.upscale_factor = upscale_factor
        self.in_channels = in_channels
        self.block_size = block_size

        self.padding_method = padding_method
        self.dtype = dtype
        self.device = device
        self.kernel_dim = kernel_dim

        # Determine reshaped channels
        self.reshaped_channels = in_channels // block_size

        # Learnable convolution after reshaping
        # self.learnable_conv = ConvLayer(
        #   in_channels=self.reshaped_channels,
        #   out_channels=self.reshaped_channels,
        #   kernel_size=self.kernel_dim * (3,),
        #   padding_mode=self.padding_method,
        #   padding=1,
        #   case = self.kernel_dim,
        #   dtype=self.dtype,
        #   device=self.device
        # )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, in_channels, depth, height, width).
        Returns:
            Tensor after learnable pixel shuffle.
        """
        batch_size, c, d, h, w = x.shape

        # Step 1: Reshape channels to spatial dimensions
        upscale = self.upscale_factor

        reshaped = x.view(
            batch_size, self.reshaped_channels, upscale, upscale, upscale, d, h, w
        )

        reshaped = reshaped.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        reshaped = reshaped.view(
            batch_size, self.reshaped_channels, d * upscale, h * upscale, w * upscale
        )

        # Step 2: Learnable convolution
        # out = self.learnable_conv(reshaped)
        # return out
        return reshaped


class TransposeConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor, dtype, device):
        super().__init__()
        self.trans_conv = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=upscale_factor,
            stride=upscale_factor,
            padding=0,
            dtype=dtype,
            device=device,
        )

    def forward(self, x):
        return self.trans_conv(x)
