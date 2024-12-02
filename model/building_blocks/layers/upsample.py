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

from utils.model_utils import reshape_jax_torch

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

  def __init__(self, 
               block_shape: Sequence[int], 
               in_channels: int,
               kernel_dim: int,
               spatial_resolution: Sequence[int]
    ):
    super(ChannelToSpace, self).__init__()

    self.block_shape = block_shape
    self.in_channels = in_channels
    self.kernel_dim = kernel_dim
    self.spatial_resolution = spatial_resolution
    self.input_dim = kernel_dim + 2 # batch size and channel dimensions are added

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
      self.spatial_resolution[i] * self.block_shape[i] for i in range(len(self.spatial_resolution))
    ]
    new_spatial_resolution = tuple(new_spatial_resolution)
    self.out_channels = self.in_channels // math.prod(self.block_shape)
    self.new_shape = (-1,) + new_spatial_resolution + (self.out_channels,)

    # Further precomputation
    batch_ndim = self.input_dim - len(self.block_shape) - 1
    # Interleave old and new spatial axes.
    spatial_axes = [i for i in range(1, 2 * len(self.block_shape) + 1)]
    reshaped = [spatial_axes[i:i + len(self.block_shape)] for i in range(0, len(spatial_axes), len(self.block_shape))]
    permuted = list(map(list, zip(*reshaped)))
    # flattened and spatial_axes is reshaped to column major row
    self.new_axes = tuple([item for sublist in permuted for item in sublist])

    # compute permutation axes:
    self.permutation_axes = tuple(range(batch_ndim)) + self.new_axes + (len(self.new_axes) + batch_ndim,)



  def forward(self, inputs: Tensor) -> Tensor:

    inputs = reshape_jax_torch(inputs, self.kernel_dim)
    x = torch.reshape(inputs, (-1,) + self.spatial_resolution + tuple(self.block_shape) + (self.out_channels,))
    x = x.permute(self.permutation_axes)
    reshaped_tensor = torch.reshape(x, self.new_shape)
    
    return reshape_jax_torch(reshaped_tensor, self.kernel_dim)