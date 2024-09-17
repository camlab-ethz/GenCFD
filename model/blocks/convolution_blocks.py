from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import default_init
from model.layers.convolutions import ConvLayer
from model.layers.residual import CombineResidualWithSkip
from model.blocks.adaptive_scaling import AdaptiveScale
from model.layers.residual import CombineResidualWithSkip

Tensor = torch.Tensor

class ResConv1x(nn.Module):
  """Single-layer residual network with size-1 conv kernels."""

  def __init__(
      self, hidden_layer_size: int, out_channels: int, act_fun: Callable[[Tensor], Tensor] = F.silu,
      dtype: torch.dtype=torch.float32, scale: float=1e-10, project_skip: bool=False):
    super(ResConv1x, self).__init__()

    self.hidden_layer_size = hidden_layer_size
    self.out_channels = out_channels
    self.act_fun = act_fun
    self.dtype = dtype
    self.scale = scale
    self.project_skip = project_skip

    self.conv1 = None
    self.conv2 = None

    self.combine_skip = CombineResidualWithSkip(project_skip=project_skip, dtype=self.dtype)
  
  def forward(self, x):

    if len(x.shape) == 4:
      kernel_size = (len(x.shape) - 2) * (1,)
      if self.conv1 is None:
        self.conv1 = nn.Conv2d(
          in_channels=self.hidden_layer_size,
          out_channels=self.hidden_layer_size,
          kernel_size=kernel_size,
          dtype=self.dtype
        )
        default_init(self.scale)(self.conv1.weight)

        self.conv2 = nn.Conv2d(
          in_channels=self.hidden_layer_size,
          out_channels=self.out_channels,
          kernel_size=kernel_size,
          dtype=self.dtype
        )
        default_init(self.scale)(self.conv2.weight)
    
    elif len(x.shape) == 5:
      kernel_size = (len(x.shape) - 2) * (1,)
      if self.conv1 is None:
        self.conv1 = nn.Conv3d(
          in_channels=self.hidden_layer_size,
          out_channels=self.hidden_layer_size,
          kernel_size=kernel_size,
          dtype=self.dtype
        )
        default_init(self.scale)(self.conv1.weight)

        self.conv2 = nn.Conv3d(
          in_channels=self.hidden_layer_size,
          out_channels=self.out_channels,
          kernel_size=kernel_size,
          dtype=self.dtype
        )
        default_init(self.scale)(self.conv2.weight)

    else:
      raise ValueError(f"Unsupported input dimension. Expected 4D or 5D")
      
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

  def __init__(self, out_channels: int, kernel_size: tuple[int, ...], 
               padding: str = 'circular', dropout: float = 0.0, 
               film_act_fun: Callable[[Tensor], Tensor] = F.silu, 
               act_fun: Callable[[Tensor], Tensor] = F.silu,
               dtype: torch.dtype = torch.float32, **kwargs):
    super(ConvBlock, self).__init__()
    
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.padding = padding
    self.dropout = dropout
    self.film_act_fun = film_act_fun
    self.act_fun = act_fun
    self.dtype = dtype

    self.norm1 = None
    self.conv1 = ConvLayer(
      features=self.out_channels,
      kernel_size=self.kernel_size,
      padding=self.padding,
      **kwargs
    )
    self.norm2 = None
    self.film = AdaptiveScale(act_fun=self.film_act_fun)
    self.dropout_layer = nn.Dropout(dropout)
    self.conv2 = ConvLayer(
      features=self.out_channels,
      kernel_size=self.kernel_size,
      padding=self.padding,
      **{'in_channels': self.out_channels}
    )
    self.res_layer = CombineResidualWithSkip(project_skip=True)

  def forward(self, x: Tensor, emb: Tensor, is_training: bool) -> Tensor:
    breakpoint()
    h = x.clone()

    if self.norm1 is None:
      # Initialize
      self.norm1 = nn.GroupNorm(min(x.shape[1] // 4, 32), x.shape[1])

    h = self.norm1(h)
    h = self.act_fun(h)
    h = self.conv1(h)

    if self.norm2 is None:
      # Initialize
      self.norm2 = nn.GroupNorm(min(h.shape[1] // 4, 32), h.shape[1])

    h = self.norm2(h)
    h = self.film(h, emb)
    h = self.act_fun(h)
    h = self.dropout_layer(h) if is_training else h
    h = self.conv2(h)
    return self.res_layer(residual=h, skip=x)



    
