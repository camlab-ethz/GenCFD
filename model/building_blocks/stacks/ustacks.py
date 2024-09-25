from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import ToPILImage, ToTensor

from model.building_blocks.layers.residual import CombineResidualWithSkip
from model.building_blocks.layers.convolutions import ConvLayer
from model.building_blocks.blocks.convolution_blocks import ConvBlock, ResConv1x
from model.building_blocks.blocks.attention_block import AttentionBlock
from utils.model_utils import channel_to_space, reshape_jax_torch

Tensor = torch.Tensor

class UStack(nn.Module):
    """Upsampling Stack.

    Takes in features at intermediate resolutions from the downsampling stack
    as well as final output, and applies upsampling with convolutional blocks
    and combines together with skip connections in typical UNet style.
    Optionally can use self attention at low spatial resolutions.

    Attributes:
        num_channels: Number of channels at each resolution level.
        num_res_blocks: Number of resnest blocks at each resolution level.
        upsample_ratio: The upsampling ration between levels.
        padding: Type of padding for the convolutional layers.
        dropout_rate: Rate for the dropout inside the transformed blocks.
        use_attention: Whether to use attention at the coarser (deepest) level.
        num_heads: Number of attentions heads inside the attention block.
        channels_per_head: Number of channels per head.
        dtype: Data type.
    """
    # TODO: Think of how to add padding tuple or int?
    def __init__(self, num_channels: tuple[int, ...], num_res_blocks: tuple[int, ...],
                 upsample_ratio: tuple[int, ...],
                 padding_method: str = 'circular', dropout_rate: float = 0.0, 
                 use_attention: bool = False, num_heads: int = 8, 
                 channels_per_head: int = -1, normalize_qk: bool = False,
                 dtype: torch.dtype=torch.float32):
        super(UStack, self).__init__()

        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.upsample_ratio = upsample_ratio
        self.padding_method = padding_method
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.num_heads = num_heads
        self.channels_per_head = channels_per_head
        self.normalize_qk = normalize_qk
        self.dtype = dtype

        # self.padding = kwargs.get('padding', 0) # TODO: MAYBE use this padding if necessary!

        self.residual_blocks = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.res_conv_blocks = nn.ModuleList()
        self.conv_layers = nn.ModuleList()

        for level, channel in enumerate(self.num_channels):
            self.conv_blocks.append(nn.ModuleList())
            self.residual_blocks.append(nn.ModuleList())

            for block_id in range(self.num_res_blocks[level]):
                self.residual_blocks[level].append(None)
                self.conv_blocks[level].append(None)

                if self.use_attention and level == 0:
                    self.attention_blocks.append(
                        AttentionBlock(num_heads=self.num_heads, normalize_qk=self.normalize_qk)
                    )
                    self.res_conv_blocks.append(
                        ResConv1x(hidden_layer_size=channel*2, out_channels=channel)
                    )
            
            self.conv_layers.append(None)

        self.res_skip_layer = None

        # add output layer
        self.conv_layers.append(None)

    def forward(self, x: Tensor, emb: Tensor, skips: list[Tensor], is_training: bool) -> Tensor:
        assert (
            len(self.num_channels)
            == len(self.num_res_blocks)
            == len(self.upsample_ratio)
        )
        kernel_dim = len(x.shape) - 2
        h = x.clone()

        for level, channel in enumerate(self.num_channels):
            for block_id in range(self.num_res_blocks[level]):
                
                if self.residual_blocks[level][block_id] is None:
                    # Initialize
                    self.residual_blocks[level][block_id] = CombineResidualWithSkip(
                        project_skip=h.shape[1] != skips[-1].shape[1]
                    )
                
                if self.conv_blocks[level][block_id] is None:
                    self.conv_blocks[level][block_id] = ConvBlock(
                        out_channels=channel,
                        kernel_size=kernel_dim * (3,),
                        padding_mode= self.padding_method,
                        **{'padding': 1, 'in_channels': h.shape[1], 'case': len(h.shape)-2}
                        )

                h = self.residual_blocks[level][block_id](residual=h, skip=skips.pop())
                h = self.conv_blocks[level][block_id](h, emb, is_training=is_training)

                if self.use_attention and level == 0:
                    h = reshape_jax_torch(h) # (bs, width, height, c)
                    b, *hw, c = h.shape
                    
                    h = self.attention_blocks[block_id](h.reshape(b, -1, c), is_training)
                    h = reshape_jax_torch(self.res_conv_blocks[block_id](reshape_jax_torch(h)).reshape(b, *hw, c))

            # upsampling
            up_ratio = self.upsample_ratio[level]
            
            if self.conv_layers[level] is None:
                self.conv_layers[level] = ConvLayer(
                    features=up_ratio**kernel_dim * channel,
                    kernel_size=kernel_dim * (3,),
                    padding_mode=self.padding_method,
                    **{'padding': 1, 'in_channels': h.shape[1], 'case': len(h.shape)-2}
                )
            h = self.conv_layers[level](h)
            h = channel_to_space(h, block_shape=kernel_dim * (up_ratio,))

        if self.res_skip_layer is None:
            self.res_skip_layer = CombineResidualWithSkip(
                project_skip=(h.shape[1] != skips[-1].shape[1]) # channel should be here!
            )
        
        if self.conv_layers[-1] is None:
            self.conv_layers[-1] = ConvLayer(
                features = 128,
                kernel_size=kernel_dim * (3,),
                padding_mode=self.padding_method,
                **{'padding': 1, 'in_channels': h.shape[1], 'case': len(h.shape)-2}
            )
        h = self.res_skip_layer(residual=h, skip=skips.pop())
        h = self.conv_layers[-1](h)
        return h


class UpsampleFourierGaussian(nn.Module):
    """Performs upsamling on input data using either the Fourier transform
    or gaussian interpolation
    """

    def __init__(self, new_shape: tuple[int, ...], 
               num_res_blocks: tuple[int, ...], mid_channel: int,
               out_channels:int, dropout_rate: int=0.0, 
               padding_method: str='circular', use_attention: bool=True,
               num_heads: int=8, dtype: torch.dtype=torch.float32,
               up_method: str='gaussian', normalize_qk: bool = False):
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
        self.up_method = up_method

        self.conv_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.res_conv_blocks = nn.ModuleList()
        self.conv_layers = None
        self.norm = None

        self.level = 0

        for i in range(self.num_res_blocks[self.level]):
            self.conv_blocks.append(None)
            self.attention_blocks.append(
                AttentionBlock(num_heads=self.num_heads, normalize_qk=normalize_qk)
                )
            self.res_conv_blocks.append(
                ResConv1x(
                    hidden_layer_size=self.mid_channel * 2,
                    out_channels=self.mid_channel
                )
            )

    def _upsample_fourier_(self, x: Tensor) -> Tensor:
        """Upsampling with the Fourier Transformation for each batch individually"""
        ndim = len(x.shape) - 2 # exlcuding the channel dim c and bs
        # N ... width, M ... height, W ... depth
        axes_map = {
            1: (2,),      # for input shape (bs, c, N)
            2: (2, 3),    # for input shape (bs, c, M, N)
            3: (2, 3, 4)  # for input shape (bs, c, W, M, N)
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
            pad_size = self.new_shape[axis] - x.size(axis)
            assert pad_size >= 0, "Padding for Upsampling can't be negative"
            pad_sizes[axis] = (pad_size // 2, pad_size - pad_size // 2)

        x_fft_padded = F.pad(x_fft, tuple(np.ravel(pad_sizes[::-1])), mode='constant')

        x_upsampled = torch.fft.ifftn(torch.fft.ifftshift(x_fft_padded, dim=axes), 
                                        dim=axes, norm='forward')
        
        return torch.real(x_upsampled)


    def _upsample_gaussian_(self, x: Tensor, method: str='linear') -> Tensor:
        """Upsampling by using Bilinear or Trilinear Interpolation"""

        ndim = len(x.shape) - 2
        if ndim == 1:
            assert len(self.new_shape) == 3, "new_shape needs to be a 3D tuple"
            size = (self.new_shape[2],)
            mode = 'linear'
        elif ndim == 2:
            assert len(self.new_shape) == 4, "new_shape needs to be a 4D tuple"
            size = (self.new_shape[2], self.new_shape[3])
            mode = 'bilinear'
        elif ndim == 3:
            assert len(self.new_shape) == 5, "new_shape needs to be a 5D tuple"
            size = (self.new_shape[2], self.new_shape[3], self.new_shape[4])
            mode = 'trilinear'
        else:
            raise ValueError("Input must be either 1D, 2D, or 3D without channel")
        
        if method == 'linear':
            return F.interpolate(x, size=size, mode=mode, align_corners=True)
        
        elif method == 'lanczos':
            assert len(x.shape) - 2 == 2, "LACZOS is only valid for a 2D grid!"
            bs, c, h, w = x.shape
            to_pil = ToPILImage()
            to_tensor = ToTensor()

            resized_img = []
            for i in range(bs):
                img_up = to_tensor(
                    TF.resize(to_pil(x[i]), size=size, interpolation=InterpolationMode.LANCZOS)
                    )
                resized_img.append(img_up)
            return torch.stack(resized_img)

        else:
            raise ValueError(
                f"For the upsampling only 'linear' and 'lanczos' interpolations are valid"
                )
      

    def forward(self, x: Tensor, emb: Tensor, is_training: bool=True) -> Tuple[Tensor, Tensor]:
        kernel_dim = len(x.shape) - 2 # number of spatial dimensions
  
        h = x.clone()
            
        for block_id in range(self.num_res_blocks[self.level]):
            if self.conv_blocks[block_id] is None:
                # Initialize
                self.conv_blocks[block_id] = ConvBlock(
                    out_channels=self.mid_channel,
                    kernel_size=kernel_dim * (3,),
                    padding_mode=self.padding_method,
                    dropout=self.dropout_rate,
                    dtype=self.dtype,
                    **{'in_channels': h.shape[1], 'padding': 1, 'case': kernel_dim}
                )
            h = self.conv_blocks[block_id](h, emb, is_training=is_training)

            if self.use_attention and self.level == 0:
                h = reshape_jax_torch(h)
                bs, *hw, c = h.shape

                h = self.attention_blocks[block_id](h.reshape(bs, -1, c), True)
                h = reshape_jax_torch(self.res_conv_blocks[block_id](reshape_jax_torch(h)).reshape(bs, *hw, c))

        if self.norm is None:
            self.norm = nn.GroupNorm(min(max(h.shape[1] // 4, 1), 32), h.shape[1])

        h = F.silu(self.norm(h))

        if self.conv_layers is None:
            # Initialize:
            self.conv_layers = ConvLayer(
                features=self.out_channels,
                kernel_size=kernel_dim * (3,),
                padding_mode=self.padding_method,
                **{'in_channels': h.shape[1], 'padding': 1, 'case': kernel_dim}
            )
        h = self.conv_layers(h)

        if self.up_method == 'fourier':
            h_up = self._upsample_fourier_(h)
        elif self.up_method == 'gaussian':
            h_up = self._upsample_gaussian_(h)
        else:
            raise ValueError(
                "Upsampling method does not exist, choose either 'fourier' or 'gaussian'" 
            )
        
        return h_up, h
    
