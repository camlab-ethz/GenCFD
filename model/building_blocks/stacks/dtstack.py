import torch
import torch.nn as nn

from model.building_blocks.layers.convolutions import ConvLayer, DownsampleConv
from model.building_blocks.blocks.convolution_blocks import ConvBlock, ResConv1x
from model.building_blocks.blocks.attention_block import AttentionBlock
from utils.model_utils import channel_to_space, reshape_jax_torch

Tensor = torch.Tensor

class DStack(nn.Module):
  """Downsampling stack.

  Repeated convolutional blocks with occasional strides for downsampling.
  Features at different resolutions are concatenated into output to use
  for skip connections by the UStack module.
  """

  def __init__(self, num_channels: tuple[int, ...], num_res_blocks: tuple[int, ...],
               downsample_ratio: tuple[int, ...], padding_method: str='circular', 
               dropout_rate: float=0.0, use_attention: bool=False, num_heads: int=8,
               channels_per_head: int=-1, use_positional_encoding: bool=False,
               normalize_qk: bool=False, dtype: torch.dtype=torch.float32):
    super(DStack, self).__init__()

    self.num_channels = num_channels
    self.num_res_blocks = num_res_blocks
    self.downsample_ratio = downsample_ratio
    self.padding_method = padding_method
    self.dropout_rate = dropout_rate
    self.use_attention = use_attention
    self.num_heads = num_heads
    self.channels_per_head = channels_per_head
    self.use_positional_encoding = use_positional_encoding
    self.dtype = dtype
    self.normalize_qk = normalize_qk

    self.conv_layer = None
    self.dsample_layers = nn.ModuleList()
    self.conv_blocks = nn.ModuleList()
    self.pos_emb_layers = nn.ModuleList()
    self.attention_blocks = nn.ModuleList()
    self.res_conv_blocks = nn.ModuleList()

    for level, channel in enumerate(self.num_channels):
      self.conv_blocks.append(nn.ModuleList())
      self.dsample_layers.append(None)

      for block_id in range(self.num_res_blocks[level]):
        self.conv_blocks[level].append(None)

        if self.use_attention and level == len(self.num_channels) - 1:
          self.pos_emb_layers.append(None)
          self.attention_blocks.append(
            AttentionBlock(num_heads=self.num_heads, normalize_qk=normalize_qk)
          )
          self.res_conv_blocks.append(
            ResConv1x(hidden_layer_size=channel*2, out_channels=channel)
          )

  def forward(self, x: Tensor, emb: Tensor, is_training: bool) -> list[Tensor]:
    assert (
      len(self.num_channels)
      == len(self.num_res_blocks)
      == len(self.downsample_ratio)
    )
    kernel_dim = len(x.shape) - 2
    skips = []
    h = x.clone()

    if self.conv_layer is None:
      self.conv_layer = ConvLayer(
        features=128,
        kernel_size=kernel_dim * (3,),
        padding_mode=self.padding_method,
        **{'padding': 1, 'in_channels': h.shape[1], 'case': kernel_dim}
      )
    
    h = self.conv_layer(h)
    skips.append(h)

    for level, channel in enumerate(self.num_channels):

      if self.dsample_layers[level] is None:
        self.dsample_layers[level] = DownsampleConv(
          features=channel,
          ratios=(self.downsample_ratio[level],) * kernel_dim,
          **{'in_channels': h.shape[1]}
        )

      h = self.dsample_layers[level](h)

      for block_id in range(self.num_res_blocks[level]):

        if self.conv_blocks[level][block_id] is None:
          self.conv_blocks[level][block_id] = ConvBlock(
            out_channels=channel,
            kernel_size=kernel_dim * (3,),
            padding_mode=self.padding_method,
            dropout=self.dropout_rate,
            **{'in_channels': h.shape[1], 'padding': 1, 'case': len(h.shape)-2}
          )
        h = self.conv_blocks[level][block_id](h, emb, is_training=is_training)

        if self.use_attention and level == len(self.num_channels) - 1:
          h = reshape_jax_torch(h) # (bs, width, height, c)
          b, *hw, c = h.shape
          h = self.attention_blocks[block_id](h.reshape(b, -1, c), is_training)
          # reshaping h first to get (bs, c, *hw), then in the end reshape again to get (bs, c, h, w)
          h = reshape_jax_torch(self.res_conv_blocks[block_id](reshape_jax_torch(h)).reshape(b, *hw, c))
        
        skips.append(h)

    return skips