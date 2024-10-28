from typing import Callable, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import default_init
from model.building_blocks.layers.convolutions import ConvLayer
from model.building_blocks.layers.residual import CombineResidualWithSkip
from model.building_blocks.blocks.adaptive_scaling import AdaptiveScale
from model.building_blocks.layers.residual import CombineResidualWithSkip

Tensor = torch.Tensor

class ResConv1x(nn.Module):
  """Single-layer residual network with size-1 conv kernels."""

  def __init__(
      self, 
      hidden_layer_size: int, 
      out_channels: int, 
      rng: torch.Generator,
      act_fun: Callable[[Tensor], Tensor] = F.silu,
      dtype: torch.dtype=torch.float32, 
      scale: float=1e-10, 
      project_skip: bool=False,
      device: Any | None = None
      ):
    super(ResConv1x, self).__init__()

    self.hidden_layer_size = hidden_layer_size
    self.out_channels = out_channels
    self.act_fun = act_fun
    self.dtype = dtype
    self.scale = scale
    self.project_skip = project_skip
    self.device = device
    self.rng = rng

    self.conv1 = None
    self.conv2 = None

    self.combine_skip = CombineResidualWithSkip(
      rng=self.rng,
      project_skip=project_skip, 
      dtype=self.dtype, 
      device=self.device
      )
  
  def forward(self, x):
    kernel_size = (len(x.shape) - 2) * (1,)

    if len(x.shape) == 3:
      # case 1
      self.conv1 = nn.Conv1d(
        in_channels=x.shape[1],
        out_channels=self.hidden_layer_size,
        kernel_size=kernel_size,
        dtype=self.dtype,
        device=self.device
      )
      self.conv2 = nn.Conv1d(
        in_channels=self.hidden_layer_size,
        out_channels=self.out_channels,
        kernel_size=kernel_size,
        dtype=self.dtype,
        device=self.device
      )

    elif len(x.shape) == 4:
      # case 2
      if self.conv1 is None:
        self.conv1 = nn.Conv2d(
          in_channels=x.shape[1],
          out_channels=self.hidden_layer_size,
          kernel_size=kernel_size,
          dtype=self.dtype,
          device=self.device
        )
        self.conv2 = nn.Conv2d(
          in_channels=self.hidden_layer_size,
          out_channels=self.out_channels,
          kernel_size=kernel_size,
          dtype=self.dtype,
          device=self.device
        )
    
    elif len(x.shape) == 5:
      # case 3
      if self.conv1 is None:
        self.conv1 = nn.Conv3d(
          in_channels=x.shape[1],
          out_channels=self.hidden_layer_size,
          kernel_size=kernel_size,
          device=self.device,
          dtype=self.dtype
        )
        self.conv2 = nn.Conv3d(
          in_channels=self.hidden_layer_size,
          out_channels=self.out_channels,
          kernel_size=kernel_size,
          device=self.device,
          dtype=self.dtype
        )

    else:
      raise ValueError(f"Unsupported input dimension. Expected 4D or 5D")
    
    default_init(self.scale)(self.conv1.weight)
    default_init(self.scale)(self.conv2.weight)

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

  def __init__(self,
               in_channels: int,
               out_channels: int, 
               kernel_size: tuple[int, ...], 
               rng: torch.Generator,
               padding_mode: str = 'circular', 
               padding: int = 0,
               stride: int = 1,
               use_bias: bool = True,
               case: int = 2,
               dropout: float = 0.0, 
               film_act_fun: Callable[[Tensor], Tensor] = F.silu, 
               act_fun: Callable[[Tensor], Tensor] = F.silu,
               dtype: torch.dtype = torch.float32,
               device: Any | None = None,
               **kwargs):
    super(ConvBlock, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.padding_mode = padding_mode
    self.dropout = dropout
    self.film_act_fun = film_act_fun
    self.act_fun = act_fun
    self.dtype = dtype
    self.device = device
    self.rng = rng
    self.padding = padding
    self.stride = stride
    self.use_bias = use_bias
    self.case = case

    self.norm1 = None
    self.conv1 = ConvLayer(
      in_channels=self.in_channels,
      out_channels=self.out_channels,
      kernel_size=self.kernel_size,
      padding_mode=self.padding_mode,
      rng = self.rng,
      padding=self.padding,
      stride=self.stride,
      use_bias=self.use_bias,
      case=self.case,
      dtype=self.dtype,
      device=self.device,
      **kwargs
    )
    self.norm2 = None
    self.film = AdaptiveScale(
      act_fun=self.film_act_fun,
      dtype=self.dtype,
      device=self.device
      )
    self.dropout_layer = nn.Dropout(dropout)
    self.conv2 = ConvLayer(
      in_channels=self.out_channels,
      out_channels=self.out_channels,
      kernel_size=self.kernel_size,
      padding_mode=self.padding_mode,
      rng=self.rng,
      padding=self.padding,
      stride=self.stride,
      use_bias=self.use_bias,
      case=self.case,
      dtype=self.dtype,
      device=self.device
    )
    self.res_layer = CombineResidualWithSkip(
      rng=self.rng,
      project_skip=True,
      dtype=self.dtype,
      device=self.device
    )

  def forward(self, x: Tensor, emb: Tensor, is_training: bool) -> Tensor:
    h = x.clone()

    if self.norm1 is None:
      # Initialize
      self.norm1 = nn.GroupNorm(
        min(max(x.shape[1] // 4, 1), 32), 
        x.shape[1],
        device=self.device,
        dtype=self.dtype
        )

    h = self.norm1(h)
    h = self.act_fun(h)
    h = self.conv1(h)

    if self.norm2 is None:
      # Initialize
      self.norm2 = nn.GroupNorm(
        min(max(h.shape[1] // 4, 1), 32), 
        h.shape[1],
        device=self.device,
        dtype=self.dtype
        )

    h = self.norm2(h)
    h = self.film(h, emb)
    h = self.act_fun(h)
    h = self.dropout_layer(h) if is_training else h
    h = self.conv2(h)
    return self.res_layer(residual=h, skip=x)