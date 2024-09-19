import torch
import torch.nn as nn

from model.layers.residual import CombineResidualWithSkip
from model.layers.convolutions import ConvLayer
from model.blocks.convolution_blocks import ConvBlock, ResConv1x
from model.blocks.attention_block import AttentionBlock
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
                    h = self.res_conv_blocks[block_id](h).reshape(b, *hw, c)
                    h = reshape_jax_torch(h) # (bs, c, h, w)

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



        







