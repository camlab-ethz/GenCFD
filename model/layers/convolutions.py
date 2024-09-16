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

"""Convolution layers."""

from typing import Literal, Sequence

import torch
from torch import nn
import torch.nn.functional as F

Tensor = torch.Tensor

def ConvLayer(
    features: int,
    kernel_size: int | Sequence[int],
    padding: str,
    use_local: bool = False,
    **kwargs
) -> nn.Module:
  """Factory for different types of convolution layers."""
  if isinstance(padding, str) and padding.lower() in ["lonlat", "latlon"]:
    if not (isinstance(kernel_size, tuple) and len(kernel_size) == 2):
      raise ValueError(
        f"kernel size {kernel_size} must be a length-2 tuple "
        f"for convolution type {padding}."
      )
    return LatLonConv(
      features, 
      kernel_size, 
      order=padding.lower(), 
      **kwargs,
    )
  
  elif use_local:
    return ConvLocal2d(
      in_channels=kwargs.get('in_channels', 1),
      out_channels=features,
      kernel_size=kernel_size,
      padding=kwargs.get('padding', 0),
      stride=kwargs.get('stride', 1),
      use_bias=kwargs.get('use_bias', True)
    )
  else:
    return nn.Conv2d(
      in_channels=kwargs.get('in_channels', 1),
      out_channels=features,
      kernel_size=kernel_size,
      padding=padding,
      stride=kwargs.get('stride', 1),
      bias=kwargs.get('use_bias', True)
    )


class ConvLocal2d(nn.Module):
  """Customized locally connected 2D convolution (ConvLocal) for PyTorch"""

  def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
               padding=0, padding_mode='constant', use_bias=True):
    super(ConvLocal2d, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    self.stride = stride if isinstance(stride, tuple) else (stride, stride)
    self.padding = padding
    self.padding_mode = padding_mode
    self.use_bias = use_bias

    # Weights for each spatial location (out_height x out_width)
    self.weights = None 

    if self.use_bias:
      self.bias = nn.Parameter(torch.zeros(out_channels))
    else:
      self.bias = None

  def forward(self, x):
    if len(x.shape) < 4:
      raise ValueError(f"Local 2D Convolution with shape length of 4 instead of {len(x.shape)}")
    
    # Input dim: (batch_size, in_channels, height, width)
    # width, height = lat, lon
    batch_size, in_channels, height, width = x.shape

    if self.padding > 0:
      x = F.pad(x, [self.padding, self.padding, self.padding, self.padding], mode=self.padding_mode, value=0)

    out_height = (height - self.kernel_size[0] + 2 * self.padding) // self.stride[0] + 1
    out_width = (width - self.kernel_size[1] + 2 * self.padding) // self.stride[1] + 1

    # Initialize weights
    if self.weights is None:
      self.weights = nn.Parameter(
        torch.empty(
          out_height, out_width, self.out_channels, in_channels, 
          self.kernel_size[0], self.kernel_size[1], device=x.device)
      )
      torch.nn.init.xavier_uniform_(self.weights)

    output = torch.zeros((batch_size, self.out_channels, out_height, out_width), device=x.device)

    # manually scripted convolution. 
    for i in range(out_height):
      for j in range(out_width):
        patch = x[:, :, i*self.stride[0]:i*self.stride[0] + self.kernel_size[0], 
                  j*self.stride[1]:j*self.stride[1] + self.kernel_size[1]]
        # Sums of the product based on Einstein's summation convention
        output[:, :, i, j] = torch.einsum('bchw, ocwh->bo', patch, self.weights[i,j])

    if self.use_bias:
      bias_shape = [1] * len(x.shape)
      bias_shape[1] = -1
      output += self.bias.view(bias_shape)

    return output


class LatLonConv(nn.Module):
  """2D convolutional layer adapted to inputs a lot-lon grid"""

  def __init__(
      self, features: int, kernel_size: tuple[int, int] = (3, 3), 
      order: Literal["latlon", "lonlat"] = "latlon", use_bias: bool = True,
      strides: tuple[int, int] = (1, 1), use_local: bool = False, **kwargs
  ):
    super(LatLonConv, self).__init__()
    self.features = features
    self.kernel_size = kernel_size
    self.order = order
    self.use_bias = use_bias
    self.strides = strides
    self.use_local = use_local

    self.in_channels = kwargs.get('in_channels', 1)

    if self.use_local:
      self.conv = ConvLocal2d(
        in_channels=self.in_channels,
        out_channels=features,
        kernel_size=kernel_size,
        stride=strides,
        bias=use_bias,
      )
    else:
      self.conv = nn.Conv2d(
        in_channels=self.in_channels,
        out_channels=features,
        kernel_size=kernel_size,
        stride=strides,
        bias=use_bias,
        padding=0,
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
      raise ValueError(f"Unrecogniized order {self.order} - 'loatlon' or 'lonlat expected.")
    
    # Circular padding to longitudinal (lon) axis
    padded_inputs = F.pad(inputs, [0, 0, lon_pad, lon_pad], mode='circular')
    # Edge padding to latitudinal (lat) axis
    padded_inputs = F.pad(padded_inputs, [lat_pad, lat_pad, 0, 0], mode='replicate')

    return self.conv(padded_inputs)


class DownsampleConv(nn.Module):
  """Downsampling layer through strided convolution."""

  def __init__(self, features: int, ratios: Sequence[int], 
               use_bias: bool = True, **kwargs):
    super(DownsampleConv, self).__init__()

    self.features = features
    self.ratios = ratios
    self.use_bias = use_bias
    self.in_channels = kwargs.get('in_channels', 1)

    # For downsampling 0 padding
    self.conv = nn.Conv2d(
      in_channels=self.in_channels,
      out_channels=features,
      kernel_size=ratios,
      bias=use_bias,
      padding=0,
    )
    
    
  def forward(self, inputs):
    """Applies strided convolution for downsampling."""
    if len(inputs.shape) <= len(self.ratios):
      raise ValueError(
        f"Inputs ({inputs.shape}) must have at least 1 more dimension " 
        f"than that of 'ratios' ({self.ratios})."
      )

    batch_ndims = len(inputs.shape) - len(self.ratios) - 1
    spatial_shape = inputs.shape[batch_ndims:-1]
    if not all(s % r == 0 for s, r in zip(spatial_shape, self.ratios)):
      raise ValueError(
        f"Input dimensions (spatial) {spatial_shape} must divide the "
        f"downsampling ratio {self.ratios}."
      )
    
    return self.conv(inputs)