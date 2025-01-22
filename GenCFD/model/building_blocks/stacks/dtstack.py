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

"""Downsampling Stack for 2D Data Dimensions"""

import torch
import torch.nn as nn
from typing import Any, Sequence

from GenCFD.model.building_blocks.layers.convolutions import ConvLayer, DownsampleConv
from GenCFD.model.building_blocks.blocks.convolution_blocks import ConvBlock, ResConv1x
from GenCFD.model.building_blocks.blocks.attention_block import AttentionBlock
from GenCFD.model.building_blocks.embeddings.position_emb import position_embedding
from GenCFD.utils.model_utils import reshape_jax_torch, default_init

Tensor = torch.Tensor


class DStack(nn.Module):
    """Downsampling stack.

    Repeated convolutional blocks with occasional strides for downsampling.
    Features at different resolutions are concatenated into output to use
    for skip connections by the UStack module.
    """

    def __init__(
        self,
        in_channels: int,
        spatial_resolution: Sequence[int],
        emb_channels: int,
        num_channels: Sequence[int],
        num_res_blocks: Sequence[int],
        downsample_ratio: Sequence[int],
        num_input_proj_channels: int = 128,
        padding_method: str = "circular",
        dropout_rate: float = 0.0,
        use_attention: bool = False,
        num_heads: int = 8,
        channels_per_head: int = -1,
        use_position_encoding: bool = False,
        normalize_qk: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Any | None = None,
    ):
        super(DStack, self).__init__()

        self.in_channels = in_channels
        self.kernel_dim = len(spatial_resolution)
        self.num_input_proj_channels = num_input_proj_channels
        self.emb_channels = emb_channels
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.downsample_ratio = downsample_ratio
        self.padding_method = padding_method
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.num_heads = num_heads
        self.channels_per_head = channels_per_head
        self.use_position_encoding = use_position_encoding
        self.dtype = dtype
        self.normalize_qk = normalize_qk
        self.device = device

        assert (
            len(self.num_channels)
            == len(self.num_res_blocks)
            == len(self.downsample_ratio)
        )

        self.conv_layer = ConvLayer(
            in_channels=self.in_channels,
            out_channels=self.num_input_proj_channels,
            kernel_size=self.kernel_dim * (3,),
            padding_mode=self.padding_method,
            padding=1,
            case=self.kernel_dim,
            kernel_init=default_init(1.0),
            dtype=self.dtype,
            device=self.device,
        )

        # Input channels for the downsampling layer
        dsample_in_channels = [self.num_input_proj_channels, *self.num_channels[:-1]]
        list_downsample_resolutions = [spatial_resolution]

        self.dsample_layers = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        self.pos_emb_layers = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.res_conv_blocks = nn.ModuleList()

        for level, channel in enumerate(self.num_channels):
            # Calculate the resolution at each level
            downsampled_resolution = tuple(
                [
                    int(res / self.downsample_ratio[level])
                    for res in list_downsample_resolutions[-1]
                ]
            )
            list_downsample_resolutions.append(downsampled_resolution)

            self.conv_blocks.append(nn.ModuleList())
            self.dsample_layers.append(
                DownsampleConv(
                    in_channels=dsample_in_channels[level],
                    out_channels=channel,
                    spatial_resolution=spatial_resolution,
                    ratios=(self.downsample_ratio[level],) * self.kernel_dim,
                    kernel_init=default_init(1.0),
                    case=self.kernel_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
            )

            for block_id in range(self.num_res_blocks[level]):
                self.conv_blocks[level].append(
                    ConvBlock(
                        in_channels=channel,
                        out_channels=channel,
                        emb_channels=self.emb_channels,
                        kernel_size=self.kernel_dim * (3,),
                        padding_mode=self.padding_method,
                        padding=1,
                        case=self.kernel_dim,
                        dropout=self.dropout_rate,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )

                if self.use_attention and level == len(self.num_channels) - 1:
                    if self.use_position_encoding:
                        self.pos_emb_layers.append(
                            position_embedding(
                                kernel_dim=self.kernel_dim,
                                in_shape=(channel,) + list_downsample_resolutions[-1],
                            ).to(self.device)
                        )
                    self.attention_blocks.append(
                        AttentionBlock(
                            in_channels=channel,
                            num_heads=self.num_heads,
                            normalize_qk=self.normalize_qk,
                            dtype=self.dtype,
                            device=self.device,
                        )
                    )
                    self.res_conv_blocks.append(
                        ResConv1x(
                            in_channels=channel,
                            hidden_layer_size=channel * 2,
                            out_channels=channel,
                            kernel_dim=1,  # 1D due to token shape being (bs, l, c)
                            dtype=self.dtype,
                            device=self.device,
                        )
                    )

    def forward(self, x: Tensor, emb: Tensor) -> list[Tensor]:

        skips = []

        h = self.conv_layer(x)
        skips.append(h)

        for level, channel in enumerate(self.num_channels):
            # Downsampling
            h = self.dsample_layers[level](h)

            for block_id in range(self.num_res_blocks[level]):
                # Convolution Block with const number of channels
                h = self.conv_blocks[level][block_id](h, emb)

                if self.use_attention and level == len(self.num_channels) - 1:
                    if self.use_position_encoding:
                        h = self.pos_emb_layers[block_id](h)
                    h = reshape_jax_torch(
                        h, kernel_dim=self.kernel_dim
                    )  # (bs, width, height, c)
                    b, *hw, c = h.shape
                    h = self.attention_blocks[block_id](h.reshape(b, -1, c))
                    # reshaping h first to get (bs, c, *hw), then in the end reshape again to get (bs, c, h, w)
                    h = reshape_jax_torch(
                        self.res_conv_blocks[block_id](
                            reshape_jax_torch(h, kernel_dim=1)
                        ),
                        kernel_dim=1,
                    ).reshape(b, *hw, c)

                    h = reshape_jax_torch(h, kernel_dim=self.kernel_dim)

                skips.append(h)

        return skips
