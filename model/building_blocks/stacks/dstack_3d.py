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

import torch
import torch.nn as nn
from typing import Any, Sequence

from model.building_blocks.layers.convolutions import ConvLayer, DownsampleConv
from model.building_blocks.blocks.convolution_blocks import ConvBlock
from model.building_blocks.blocks.attention_block import AxialSelfAttentionBlock
from utils.model_utils import default_init

Tensor = torch.Tensor

class DStack(nn.Module):
  """Downsampling stack.

  Repeated convolutional blocks with occasional strides for downsampling.
  Features at different resolutions are concatenated into output to use
  for skip connections by the UStack module.
  """

  def __init__(self, 
               num_channels: tuple[int, ...], 
               num_res_blocks: tuple[int, ...],
               downsample_ratio: tuple[int, ...], 
               use_spatial_attention: Sequence[bool],
               rng: torch.Generator,
               num_input_proj_channels: int = 128,
               padding_method: str='circular', # LATLON
               dropout_rate: float=0.0, 
               num_heads: int=8,
               channels_per_head: int=-1, 
               use_position_encoding: bool=False,
               normalize_qk: bool=False, 
               dtype: torch.dtype=torch.float32,
               device: Any | None = None):
    super(DStack, self).__init__()

    self.num_channels = num_channels
    self.num_res_blocks = num_res_blocks
    self.downsample_ratio = downsample_ratio
    self.padding_method = padding_method
    self.dropout_rate = dropout_rate
    self.use_spatial_attention = use_spatial_attention
    self.num_input_proj_channels = num_input_proj_channels
    self.num_heads = num_heads
    self.channels_per_head = channels_per_head
    self.use_position_encoding = use_position_encoding
    self.normalize_qk = normalize_qk
    self.dtype = dtype
    self.device = device
    self.rng = rng

    self.conv_layer = None # ConvLayer
    self.dsample_layers = nn.ModuleList() # DownsampleConv layer
    self.conv_blocks = nn.ModuleList() # ConvBlock
    self.attention_blocks = nn.ModuleList() # AxialSelfAttentionBlock

    for level, channel in enumerate(self.num_channels):
      self.conv_blocks.append(nn.ModuleList())
      self.attention_blocks.append(nn.ModuleList())
      self.dsample_layers.append(None)

      for block_id in range(self.num_res_blocks[level]):
        self.conv_blocks[level].append(None)

        if self.use_spatial_attention[level]:
            # attention requires input shape: (bs, x, y, z, c)
            attn_axes = [1, 2, 3] # attention along all spatial dimensions
            
            self.attention_blocks[level].append(
                AxialSelfAttentionBlock(
                rng=self.rng,
                attention_axes=attn_axes,
                add_position_embedding=self.use_position_encoding,
                num_heads=self.num_heads,
                dtype=self.dtype,
                device=self.device
                )
            )

  def forward(self, x: Tensor, emb: Tensor, is_training: bool) -> list[Tensor]:
    assert (x.ndim == 5) # TODO: add or statement if time is included
    assert x.shape[0] == emb.shape[0]
    assert len(self.num_channels) == len(self.num_res_blocks)
    assert len(self.downsample_ratio) == len(self.num_res_blocks)

    kernel_dim = len(x.shape) - 2
    skips = []
    h = x.clone()

    if self.conv_layer is None:
      self.conv_layer = ConvLayer(
        in_channels=h.shape[1],
        out_channels=self.num_input_proj_channels,
        kernel_size=kernel_dim * (3,),
        padding_mode=self.padding_method,
        rng=self.rng,
        padding=1,
        case=kernel_dim,
        kernel_init=default_init(1.0),
        dtype=self.dtype,
        device=self.device,
      )
    
    h = self.conv_layer(h)
    skips.append(h)

    for level, channel in enumerate(self.num_channels):

        if self.dsample_layers[level] is None:
            self.dsample_layers[level] = DownsampleConv(
                in_channels=h.shape[1],
                out_channels=channel,
                ratios=(self.downsample_ratio[level],) * kernel_dim,
                kernel_init=default_init(1.0),
                rng=self.rng,
                device=self.device,
                dtype=self.dtype,
            )

        h = self.dsample_layers[level](h)

        for block_id in range(self.num_res_blocks[level]):

            if self.conv_blocks[level][block_id] is None:
                self.conv_blocks[level][block_id] = ConvBlock(
                    in_channels=h.shape[1],
                    out_channels=channel,
                    kernel_size=kernel_dim * (3,),
                    rng=self.rng,
                    padding_mode=self.padding_method,
                    padding=1,
                    case=len(h.shape)-2,
                    dropout=self.dropout_rate,
                    dtype=self.dtype,
                    device=self.device,
                )
            h = self.conv_blocks[level][block_id](h, emb, is_training=is_training)

            if self.use_spatial_attention[level]:  
                h = self.attention_blocks[level][block_id](h, is_training)
            
            skips.append(h)

    return skips