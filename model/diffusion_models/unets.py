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

"""U-Net denoiser models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from model.layers.resize import FilteredResize
from model.stacks.dtstack import DStack
from model.stacks.ustacks import UpsampleFourierGaussian, UStack
from model.embeddings.fourier_emb import FourierEmbedding
from model.layers.residual import CombineResidualWithSkip
from model.layers.convolutions import ConvLayer
from utils.model_utils import reshape_jax_torch


Tensor = torch.Tensor


class UNet(nn.Module):
  """UNet model compatible with 1 or 2 spatial dimensions.

  Original UNet model transformed from a Jax based to a PyTorch
  based version. Derived from Wan et al. (https://arxiv.org/abs/2305.15618)
  """

  def __init__(self, out_channels: int, resize_to_shape: tuple[int, ...] | None = None,
               use_hr_residual: bool = False,
               num_channels: tuple[int, ...] = (128, 256, 256, 256),
               downsample_ratio : tuple[int, ...] = (2, 2, 2, 2), num_blocks: int = 4, 
               noise_embed_dim: int = 128, padding_method: str = 'circular', 
               dropout_rate: float = 0.0, use_attention: bool = True, 
               use_position_encoding: bool = True, num_heads: int = 8,
               normalize_qk: bool = False, dtype: torch.dtype = torch.float32):
    super(UNet, self).__init__()

    self.out_channels = out_channels
    self.resize_to_shape = resize_to_shape
    self.num_channels = num_channels
    self.downsample_ratio = downsample_ratio
    self.use_hr_residual = use_hr_residual
    self.num_blocks = num_blocks
    self.noise_embed_dim = noise_embed_dim
    self.padding_method = padding_method
    self.dropout_rate = dropout_rate
    self.use_attention = use_attention
    self.use_position_encoding = use_position_encoding
    self.num_heads = num_heads
    self.normalize_qk = normalize_qk
    self.dtype = dtype

    self.embedding = FourierEmbedding(
      dims=self.noise_embed_dim,
      dtype=self.dtype
    )
    self.DStack = DStack(
      num_channels=self.num_channels,
      num_res_blocks=len(self.num_channels) * (self.num_blocks,),
      downsample_ratio=self.downsample_ratio,
      padding_method=self.padding_method,
      dropout_rate=self.dropout_rate,
      use_attention=self.use_attention,
      num_heads=self.num_heads,
      use_positional_encoding=self.use_position_encoding,
      normalize_qk=self.normalize_qk,
      dtype=self.dtype
    )
    self.UStack = UStack(
      num_channels=self.num_channels[::-1],
      num_res_blocks=len(self.num_channels) * (self.num_blocks,),
      upsample_ratio=self.downsample_ratio[::-1],
      padding_method=self.padding_method,
      dropout_rate=self.dropout_rate,
      use_attention=self.use_attention,
      num_heads=self.num_heads,
      normalize_qk=self.normalize_qk,
      dtype=self.dtype
    )
    self.norm = None
    self.conv_layer = None
    
    if self.use_hr_residual:
      self.upsample = UpsampleFourierGaussian(
        new_shape=self.num_channels[::-1],
        num_res_blocks=len(self.num_channels) * (self.num_blocks,),
        mid_channel=256,
        out_channels=self.out_channels,
        padding_method=self.padding_method,
        dropout_rate=self.dropout_rate,
        use_attention=self.use_attention,
        num_heads=self.num_heads,
        normalize_qk=self.normalize_qk
      )
      self.res_skip = None

  def forward(self, x: Tensor, sigma: Tensor, is_training: bool = True, 
                down_only: bool = False) -> Tensor:
    """Predicts denosied given noise input and noise level.
    
    Args:
      x: The model input (i.e. noise sample) with shape (bs, **spatial_dims, c)
      sigma: The noise level, which either shares the same bs dim as 'x' 
              or is a scalar
      is_training: A flag that indicates whether the module runs in training mode.
      down_only: If set to 'True', only returns 'skips[-1]' (used for downstream
                  tasks) as an embedding. If set to 'False' it then does the full
                  UNet usual computation.
    
    Returns:
      An output tensor with the same dimension as 'x'.
    """

    if sigma.dim() < 1:
      sigma = sigma.expand(x.size(0))

    if sigma.dim() != 1 or x.shape[0] != sigma.shape[0]:
      raise ValueError(
        "sigma must be 1D and have the same leading (batch) dim as x"
        f" ({x.shape[0]})"
      )

    kernel_dim = x.dim() - 2

    emb = self.embedding(sigma)
    skips = self.DStack(x, emb, is_training=is_training)

    if down_only:
      return skips[-1]
    
    if self.use_hr_residual:
      # high_res_residual = 0
      high_res_residual, _ = self.upsample(skips[-1], emb, is_training=is_training)
    
    h = self.UStack(skips[-1], emb, skips, is_training=is_training)

    if self.norm is None:
      self.norm = nn.GroupNorm(min(max(h.shape[1] // 4, 1), 32), h.shape[1])

    h = F.silu(self.norm(h))

    if self.conv_layer is None:
      self.conv_layer = ConvLayer(
        features=self.out_channels,
        kernel_size=kernel_dim * (3,),
        padding_mode=self.padding_method,
        **{'in_channels': h.shape[1], 'padding': 1, 'case': kernel_dim}
      )
    
    h = self.conv_layer(h)

    if self.use_hr_residual:
      if self.res_skip is None:
        # TODO: There is a mismatch
        # when it comes to the spatial dimension!
        self.res_skip = CombineResidualWithSkip(
          project_skip=not(h.shape[1] == high_res_residual.shape[1]),
          dtype = self.dtype
        )
      h = self.res_skip(residual=h, skip=high_res_residual)
    
    return h