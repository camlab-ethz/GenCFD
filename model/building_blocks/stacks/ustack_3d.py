from typing import Any, Sequence
import torch
import torch.nn as nn
from model.building_blocks.layers.residual import CombineResidualWithSkip
from model.building_blocks.layers.convolutions import ConvLayer
from model.building_blocks.blocks.convolution_blocks import ConvBlock
from model.building_blocks.blocks.attention_block import AxialSelfAttentionBlock
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
    def __init__(self, 
                 num_channels: tuple[int, ...], 
                 num_res_blocks: tuple[int, ...],
                 upsample_ratio: tuple[int, ...],
                 use_spatial_attention: Sequence[bool],
                 rng: torch.Generator,
                 num_output_proj_channels: int = 128,
                 padding_method: str = 'circular', 
                 dropout_rate: float = 0.0, 
                 num_heads: int = 8, 
                 channels_per_head: int = -1, 
                 normalize_qk: bool = False,
                 use_position_encoding: bool = False,
                 dtype: torch.dtype=torch.float32,
                 device: Any | None = None):
        super(UStack, self).__init__()

        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.upsample_ratio = upsample_ratio
        self.padding_method = padding_method
        self.dropout_rate = dropout_rate
        self.use_spatial_attention = use_spatial_attention
        self.num_output_proj_channels = num_output_proj_channels
        self.num_heads = num_heads
        self.channels_per_head = channels_per_head
        self.normalize_qk = normalize_qk
        self.use_position_encoding = use_position_encoding
        self.dtype = dtype
        self.device = device
        self.rng = rng

        self.residual_blocks = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.conv_layers = nn.ModuleList()

        for level, channel in enumerate(self.num_channels):
            self.conv_blocks.append(nn.ModuleList())
            self.residual_blocks.append(nn.ModuleList())

            for block_id in range(self.num_res_blocks[level]):
                self.residual_blocks[level].append(None)
                self.conv_blocks[level].append(None)

                if self.use_spatial_attention:
                    # attention requires input shape: (bs, x, y, z, c)
                    attn_axes = [1, 2, 3] # attention along all spatial dimensions

                    self.attention_blocks.append(
                        AxialSelfAttentionBlock(
                            rng=self.rng,
                            attention_axes=attn_axes,
                            add_position_embedding=self.use_position_encoding,
                            num_heads=self.num_heads,
                            dtype=self.dtype,
                            device=self.device
                        )
                    )
            
            self.conv_layers.append(None)

        self.res_skip_layer = None

        # add output layer
        self.conv_layers.append(None)

    def forward(self, x: Tensor, emb: Tensor, skips: list[Tensor], is_training: bool) -> Tensor:
        assert x.ndim == 5
        assert x.shape[0] == emb.shape[0]
        assert len(self.num_channels) == len(self.num_res_blocks)
        assert len(self.upsample_ratio) == len(self.num_res_blocks)

        kernel_dim = len(x.shape) - 2
        h = x.clone()

        for level, channel in enumerate(self.num_channels):
            for block_id in range(self.num_res_blocks[level]):
                
                if self.residual_blocks[level][block_id] is None:
                    self.residual_blocks[level][block_id] = CombineResidualWithSkip(
                        rng=self.rng,
                        project_skip=h.shape[1] != skips[-1].shape[1],
                        dtype=self.dtype,
                        device=self.device
                    )
                
                h = self.residual_blocks[level][block_id](residual=h, skip=skips.pop())
                
                if self.conv_blocks[level][block_id] is None:
                    self.conv_blocks[level][block_id] = ConvBlock(
                        in_channels = h.shape[1],
                        out_channels=channel,
                        kernel_size=kernel_dim * (3,),
                        rng=self.rng,
                        padding_mode= self.padding_method,
                        padding=1,
                        case=len(h.shape)-2,
                        dtype=self.dtype,
                        device=self.device,
                        )

                h = self.conv_blocks[level][block_id](h, emb, is_training=is_training)

                if self.use_spatial_attention[level]:
                    h = self.attention_blocks[block_id](h, is_training)

            # upsampling
            up_ratio = self.upsample_ratio[level]
            
            if self.conv_layers[level] is None:
                self.conv_layers[level] = ConvLayer(
                    in_channels=h.shape[1],
                    out_channels=up_ratio**kernel_dim * channel,
                    kernel_size=kernel_dim * (3,),
                    padding_mode=self.padding_method,
                    rng=self.rng,
                    padding=1,
                    case=len(h.shape)-2,
                    dtype=self.dtype,
                    device=self.device
                )
            h = self.conv_layers[level](h)
            h = channel_to_space(h, block_shape=kernel_dim * (up_ratio,))

        if self.res_skip_layer is None:
            self.res_skip_layer = CombineResidualWithSkip(
                rng=self.rng,
                project_skip=(h.shape[1] != skips[-1].shape[1]), # channel should be here!
                dtype=self.dtype,
                device=self.device
            )
        h = self.res_skip_layer(residual=h, skip=skips.pop())
        
        if self.conv_layers[-1] is None:
            self.conv_layers[-1] = ConvLayer(
                in_channels=h.shape[1],
                out_channels=self.num_output_proj_channels,
                kernel_size=kernel_dim * (3,),
                padding_mode=self.padding_method,
                rng=self.rng,
                padding=1,
                case=len(h.shape)-2,
                dtype=self.dtype,
                device=self.device,
            )
        h = self.conv_layers[-1](h)
        return h