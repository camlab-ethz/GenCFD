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

"""U-Net denoiser models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Sequence

from utils.model_utils import default_init
from model.building_blocks.stacks.dtstack import DStack
from model.building_blocks.stacks.ustacks import UpsampleFourierGaussian, UStack
from model.building_blocks.embeddings.fourier_emb import FourierEmbedding
from model.building_blocks.layers.residual import CombineResidualWithSkip
from model.building_blocks.layers.convolutions import ConvLayer

Tensor = torch.Tensor


class UNet(nn.Module):
  """UNet model compatible with 1 or 2 spatial dimensions.

  Original UNet model transformed from a Jax based to a PyTorch
  based version. Derived from Wan et al. (https://arxiv.org/abs/2305.15618)
  """

  def __init__(
      self, 
      in_channels: int,
      out_channels: int,
      spatial_resolution: Sequence[int],
      time_cond: bool,
      resize_to_shape: Sequence[int] | None = None,
      use_hr_residual: bool = False,
      num_channels: Sequence[int] = (128, 256, 256),
      downsample_ratio : Sequence[int] = (2, 2, 2), 
      num_blocks: int = 4, 
      noise_embed_dim: int = 128, 
      input_proj_channels: int = 128,
      output_proj_channels: int = 128,
      padding_method: str = 'circular', 
      dropout_rate: float = 0.0, 
      use_attention: bool = True, 
      use_position_encoding: bool = True, 
      num_heads: int = 8,
      normalize_qk: bool = False, 
      dtype: torch.dtype = torch.float32, 
      device: Any | None = None,
      buffer_dict: dict = None
    ):
    super(UNet, self).__init__()

    if buffer_dict:
        # Store normalization parameters as buffers for all datasets!
        for name, tensor in buffer_dict.items():
          self.register_buffer(name, tensor)

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.spatial_resolution = spatial_resolution
    self.time_cond = time_cond # can be used if additional conditioning on time is required
    self.kernel_dim = len(spatial_resolution)
    # resize_to_shape can be utilized if the dataset resolution changes within batches
    self.resize_to_shape = resize_to_shape
    self.num_channels = num_channels
    self.downsample_ratio = downsample_ratio
    self.use_hr_residual = use_hr_residual
    self.num_blocks = num_blocks
    self.noise_embed_dim = noise_embed_dim
    self.input_proj_channels = input_proj_channels
    self.output_proj_channels = output_proj_channels
    self.padding_method = padding_method
    self.dropout_rate = dropout_rate
    self.use_attention = use_attention
    self.use_position_encoding = use_position_encoding
    self.num_heads = num_heads
    self.normalize_qk = normalize_qk
    self.dtype = dtype
    self.device = device

    self.embedding = FourierEmbedding(
      dims=self.noise_embed_dim,
      dtype=self.dtype,
      device=self.device
    )

    self.emb_channels = self.noise_embed_dim * 2 if self.time_cond else self.noise_embed_dim

    self.DStack = DStack(
      in_channels=self.in_channels,
      spatial_resolution=self.spatial_resolution,
      emb_channels=self.emb_channels,
      num_channels=self.num_channels,
      num_res_blocks=len(self.num_channels) * (self.num_blocks,),
      downsample_ratio=self.downsample_ratio,
      num_input_proj_channels=self.input_proj_channels,
      padding_method=self.padding_method,
      dropout_rate=self.dropout_rate,
      use_attention=self.use_attention,
      num_heads=self.num_heads,
      use_position_encoding=self.use_position_encoding,
      normalize_qk=self.normalize_qk,
      dtype=self.dtype,
      device=self.device
    )

    self.UStack = UStack(
      spatial_resolution=self.spatial_resolution,
      emb_channels=self.emb_channels,
      num_channels=self.num_channels[::-1],
      num_res_blocks=len(self.num_channels) * (self.num_blocks,),
      upsample_ratio=self.downsample_ratio[::-1],
      padding_method=self.padding_method,
      dropout_rate=self.dropout_rate,
      use_attention=self.use_attention,
      num_input_proj_channels=self.input_proj_channels,
      num_output_proj_channels=self.output_proj_channels,
      num_heads=self.num_heads,
      normalize_qk=self.normalize_qk,
      dtype=self.dtype,
      device=self.device
    )

    self.norm = nn.GroupNorm(
      min(max(self.output_proj_channels // 4, 1), 32), 
      self.output_proj_channels,
      device=self.device,
      dtype=self.dtype
    )

    self.conv_layer = ConvLayer(
      in_channels=self.output_proj_channels,
      out_channels=self.out_channels,
      kernel_size=self.kernel_dim * (3,),
      padding_mode=self.padding_method,
      padding=1,
      case=self.kernel_dim,
      kernel_init=default_init(),
      dtype=self.dtype,
      device=self.device
    )
    
    if self.use_hr_residual:
      self.upsample = None
      self.res_skip = None

  def forward(
      self, 
      x: Tensor, 
      sigma: Tensor, 
      time: Tensor = None, 
      down_only: bool = False
    ) -> Tensor:
    """Predicts denosied given noise input and noise level.
    
    Args:
      x: The model input (i.e. noise sample) with shape (bs, **spatial_dims, c)
      sigma: The noise level, which either shares the same bs dim as 'x' 
              or is a scalar
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

    # Downsampling
    skips = self.DStack(x, emb)

    if down_only:
      return skips[-1]
    
    if self.use_hr_residual:
      if self.upsample is None:
        self.upsample = UpsampleFourierGaussian(
        new_shape=x.shape,
        num_res_blocks=len(self.num_channels) * (self.num_blocks,),
        mid_channel=256,
        out_channels=self.out_channels,
        emb_channels=self.emb_channels,
        padding_method=self.padding_method,
        dropout_rate=self.dropout_rate,
        use_attention=self.use_attention,
        num_heads=self.num_heads,
        dtype=self.dtype,
        device=self.device,
        normalize_qk=self.normalize_qk
      )
      high_res_residual, _ = self.upsample(skips[-1], emb)
    
    # Upsampling
    h = self.UStack(skips[-1], emb, skips)

    h = F.silu(self.norm(h))
    h = self.conv_layer(h)

    if self.use_hr_residual:
      if self.res_skip is None:
        self.res_skip = CombineResidualWithSkip(
          residual_channels=h.shape[1],
          skip_channels=high_res_residual,
          kernel_dim=self.kernel_dim,
          project_skip=not(h.shape[1] == high_res_residual.shape[1]),
          dtype = self.dtype,
          device=self.device
        )
      h = self.res_skip(residual=h, skip=high_res_residual)
    
    return h
  

class PreconditionedDenoiser(UNet):
  """Preconditioned denoising model."""

  def __init__(
      self, 
      in_channels: int,
      out_channels: int,
      spatial_resolution: Sequence[int],
      time_cond: bool,
      resize_to_shape: tuple[int, ...] | None = None,
      use_hr_residual: bool = False,
      num_channels: tuple[int, ...] = (128, 256, 256),
      downsample_ratio : tuple[int, ...] = (2, 2, 2), 
      num_blocks: int = 4, 
      noise_embed_dim: int = 128, 
      input_proj_channels: int = 128,
      output_proj_channels: int = 128,
      padding_method: str = 'circular', 
      dropout_rate: float = 0.0, 
      use_attention: bool = True, 
      use_position_encoding: bool = True, 
      num_heads: int = 8,
      normalize_qk: bool = False, 
      dtype: torch.dtype = torch.float32, 
      device: Any | None = None,
      buffer_dict: dict = None,
      sigma_data: float = 1.0
    ):
    super().__init__(
      in_channels=in_channels,
      out_channels=out_channels,
      spatial_resolution=spatial_resolution,
      time_cond=time_cond,
      resize_to_shape=resize_to_shape,
      use_hr_residual=use_hr_residual,
      num_channels=num_channels,
      downsample_ratio=downsample_ratio,
      num_blocks=num_blocks,
      noise_embed_dim=noise_embed_dim,
      input_proj_channels=input_proj_channels,
      output_proj_channels=output_proj_channels,
      padding_method=padding_method,
      dropout_rate=dropout_rate,
      use_attention=use_attention,
      use_position_encoding=use_position_encoding,
      num_heads=num_heads,
      normalize_qk=normalize_qk,
      dtype=dtype,
      device=device,
      buffer_dict=buffer_dict
    )
    
    self.sigma_data = sigma_data
    

  def forward(
      self, 
      x: Tensor, 
      sigma: Tensor,
      y: Tensor = None,
      time: Tensor = None,
      down_only: bool = False, 
    ) -> Tensor:
    """Runs preconditioned denoising."""
    if sigma.dim() < 1:
      sigma = sigma.expand(x.shape[0])
    
    if sigma.dim() != 1 or x.shape[0] != sigma.shape[0]:
      raise ValueError(
        "sigma must be 1D and have the same leading (batch) dim as x"
        f" ({x.shape[0]})"
      )
    
    total_var = self.sigma_data ** 2 + sigma ** 2
    c_skip = self.sigma_data ** 2 / total_var
    c_out = sigma * self.sigma_data / torch.sqrt(total_var)
    c_in = 1 / torch.sqrt(total_var)
    c_noise = 0.25 * torch.log(sigma)

    expand_shape = [-1] + [1] * (x.dim() - 1)
    # Expand dimensions of the coefficients
    c_in = c_in.view(*expand_shape)
    c_out = c_out.view(*expand_shape)
    c_skip = c_skip.view(*expand_shape)

    inputs = c_in * x
    if y is not None:
      # stack conditioning y
      inputs = torch.cat((inputs, y), dim=1)

    f_x = super().forward(
      inputs,
      sigma=c_noise,
      time=time,
      down_only=down_only
    )
 
    if down_only:
      return f_x
    
    return c_skip * x + c_out * f_x