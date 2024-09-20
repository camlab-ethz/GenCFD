# Copyright 2024 The swirl_dynamics Authors and CAM Lab at ETH Zurich.
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
# Converted from JAX to PyTorch and created the UpsampleFourier method.

"""Upsampling modules."""

from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.model_utils import reshape_jax_torch

Tensor = torch.Tensor


def channel_to_space(inputs: Tensor, block_shape: Sequence[int]) -> Tensor:
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

  Returns:
    The upsampled array.
  """
  # reshape from (bs, c, y, x) to (bs, x, y, c)
  inputs = reshape_jax_torch(inputs)
  if not len(inputs.shape) > len(block_shape):
    raise ValueError(
        f"Ndim of `x` ({len(inputs.shape)}) expected to be higher than the length of"
        f" `block_shape` {len(block_shape)}."
    )

  if inputs.shape[-1] % np.prod(block_shape) != 0:
    raise ValueError(
        f"The number of channels in the input ({inputs.shape[-1]}) must be"
        f" divisible by the block size ({np.prod(block_shape)})."
    )

  # Create additional spatial axes from channel.
  old_shape = inputs.shape
  batch_ndim = inputs.ndim - len(block_shape) - 1
  cout = old_shape[-1] // np.prod(block_shape)
  x = torch.reshape(inputs, old_shape[:-1] + tuple(block_shape) + (cout,))

  # Interleave old and new spatial axes.
  spatial_axes = np.arange(2 * len(block_shape), dtype=np.int32) + batch_ndim
  new_axes = spatial_axes.reshape(2, -1).ravel(order="F")
  x = torch.permute(
      x,
      tuple(range(batch_ndim))
      + tuple(new_axes)
      + (len(new_axes) + batch_ndim,),
  )

  # Merge the interleaved axes.
  new_shape = np.asarray(old_shape[batch_ndim:-1]) * np.asarray(block_shape)
  new_shape = old_shape[:batch_ndim] + tuple(new_shape) + (cout,)
  return reshape_jax_torch(torch.reshape(x, new_shape))

class UpsampleFourierGaussian(nn.module):
  """Performs upsamling on input data using the Fourier transform"""

  def __init__(self, new_shape: tuple[int, ...], 
               num_res_blocks: tuple[int, ...], mid_channel: int,
               dropout_rate: int, out_channels:int, 
               padding_method: str='circular', use_attention: bool=True,
               num_heads: int=8, dtype: torch.dtype=torch.float32):
    super(UpsampleFourierGaussian, self).__init__()

    self.new_shape = new_shape
    self.num_res_blocks = num_res_blocks
    self.mid_channel = mid_channel
    self.dropout_rate = dropout_rate
    self.out_channels = out_channels
    self.padding_method = padding_method
    self.use_attention = use_attention
    self.num_heads = num_heads
    self.dtype = dtype

    def _upsample_fourier_(self, x: Tensor) -> Tensor:
      """Upsampling with the Fourier Transformation"""
      ndim = len(x.shape) - 1 # exlcuding the channel dim c
      # N ... width, M ... height, W ... depth
      axes_map = {
        1: (1,),      # for input shape (c, N)
        2: (1, 2),    # for input shape (c, M, N)
        3: (1, 2, 3)  # for input shape (c, W, M, N)
      }
      if ndim not in axes_map:
        raise ValueError(
          "Input must be either 2D, 3D or 4D with including the channel dim"
          )
      
      axes = axes_map[ndim]
      
      x_fft = torch.fft.fftshift(torch.fft.fftn(x, dim=axes, norm='forward'), 
                                 dim=axes)
      
      pad_sizes = [(0, 0)] * len(x.shape)
      for axis in axes:
        pad_size = self.new_shape[axis - 2] - x.size(axis)
        pad_sizes[axis] = (pad_size // 2, pad_size - pad_size // 2)

      x_fft_padded = F.pad(x_fft, tuple(np.ravel(pad_sizes[::-1])), model='constant')

      x_upsampled = torch.fft.ifftn(torch.fft.ifftshift(x_fft_padded, dim=axes), 
                                    dim=axes, norm='forward')
      
      return torch.real(x_upsampled)


    def _upsample_gaussian_(self, x: Tensor) -> Tensor:
      """Upsampling by using Bilinear or Trilinear Interpolation"""
      pass

    def forward(self, x:Tensor, emb: Tensor, is_training: bool=True) -> Tensor:
      pass

