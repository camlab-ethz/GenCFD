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

"""Upsampling Stack for 2D Data Dimensions"""

from typing import Tuple, Any, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import ToPILImage, ToTensor

from GenCFD.model.building_blocks.layers.residual import CombineResidualWithSkip
from GenCFD.model.building_blocks.layers.convolutions import ConvLayer
from GenCFD.model.building_blocks.blocks.convolution_blocks import ConvBlock, ResConv1x
from GenCFD.model.building_blocks.blocks.attention_block import AttentionBlock
from GenCFD.utils.model_utils import reshape_jax_torch, default_init

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

    def __init__(
        self,
        spatial_resolution: Sequence[int],
        emb_channels: int,
        num_channels: Sequence[int],
        num_res_blocks: Sequence[int],
        upsample_ratio: Sequence[int],
        padding_method: str = "circular",
        dropout_rate: float = 0.0,
        use_attention: bool = False,
        num_input_proj_channels: int = 128,
        num_output_proj_channels: int = 128,
        num_heads: int = 8,
        channels_per_head: int = -1,
        normalize_qk: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ):
        super(UStack, self).__init__()

        self.kernel_dim = len(spatial_resolution)
        self.emb_channels = emb_channels
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.upsample_ratio = upsample_ratio
        self.padding_method = padding_method
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.num_input_proj_channels = num_input_proj_channels
        self.num_output_proj_channels = num_output_proj_channels
        self.num_heads = num_heads
        self.channels_per_head = channels_per_head
        self.normalize_qk = normalize_qk
        self.dtype = dtype
        self.device = device

        # Calculate channels for the residual block
        in_channels = []

        # calculate list of upsample resolutions
        list_upsample_resolutions = [spatial_resolution]
        for level, channel in enumerate(self.num_channels):
            downsampled_resolution = tuple(
                [
                    int(res / self.upsample_ratio[level])
                    for res in list_upsample_resolutions[-1]
                ]
            )
            list_upsample_resolutions.append(downsampled_resolution)
        list_upsample_resolutions = list_upsample_resolutions[::-1]
        list_upsample_resolutions.pop()

        self.residual_blocks = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.res_conv_blocks = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()

        for level, channel in enumerate(self.num_channels):
            self.conv_blocks.append(nn.ModuleList())
            self.residual_blocks.append(nn.ModuleList())

            for block_id in range(self.num_res_blocks[level]):
                if block_id == 0 and level > 0:
                    in_channels.append(self.num_channels[level - 1])
                else:
                    in_channels.append(channel)

                self.residual_blocks[level].append(
                    CombineResidualWithSkip(
                        residual_channels=in_channels[-1],
                        skip_channels=channel,
                        kernel_dim=self.kernel_dim,
                        project_skip=in_channels[-1] != channel,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )
                self.conv_blocks[level].append(
                    ConvBlock(
                        in_channels=in_channels[-1],
                        out_channels=channel,
                        emb_channels=self.emb_channels,
                        kernel_size=self.kernel_dim * (3,),
                        padding_mode=self.padding_method,
                        padding=1,
                        case=self.kernel_dim,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )

                if self.use_attention and level == 0:
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
                            kernel_dim=1,  # 1D due to token shape (bs, l, c)
                            dtype=self.dtype,
                            device=self.device,
                        )
                    )

            # Upsampling step
            up_ratio = self.upsample_ratio[level]
            self.conv_layers.append(
                ConvLayer(
                    in_channels=channel,
                    out_channels=up_ratio**self.kernel_dim * channel,
                    kernel_size=self.kernel_dim * (3,),
                    padding_mode=self.padding_method,
                    padding=1,
                    case=self.kernel_dim,
                    kernel_init=default_init(1.0),
                    dtype=self.dtype,
                    device=self.device,
                )
            )

            # For higher or lower input dimensions than 2 use ChannelToSpace from upsample.py
            self.upsample_layers.append(
                # upscaling spatial dimensions by rearranging channel data
                nn.PixelShuffle(up_ratio)  # Only for 2D data input
            )

        # DStack Input - UStack Output Residual Connection
        self.res_skip_layer = CombineResidualWithSkip(
            residual_channels=self.num_channels[-1],
            skip_channels=self.num_input_proj_channels,
            project_skip=(self.num_channels[-1] != self.num_input_proj_channels),
            dtype=self.dtype,
            device=self.device,
        )

        # Add output layer
        self.conv_layers.append(
            ConvLayer(
                in_channels=self.num_channels[-1],
                out_channels=self.num_output_proj_channels,
                kernel_size=self.kernel_dim * (3,),
                padding_mode=self.padding_method,
                padding=1,
                case=self.kernel_dim,
                kernel_init=default_init(1.0),
                dtype=self.dtype,
                device=self.device,
            )
        )

    def forward(self, x: Tensor, emb: Tensor, skips: list[Tensor]) -> Tensor:
        assert (
            len(self.num_channels)
            == len(self.num_res_blocks)
            == len(self.upsample_ratio)
        )
        h = x

        for level, channel in enumerate(self.num_channels):
            for block_id in range(self.num_res_blocks[level]):
                # Residual
                h = self.residual_blocks[level][block_id](residual=h, skip=skips.pop())
                # Convolution Blocks
                h = self.conv_blocks[level][block_id](h, emb)
                # Spatial Attention Blocks
                if self.use_attention and level == 0:
                    h = reshape_jax_torch(
                        h, kernel_dim=self.kernel_dim
                    )  # (bs, width, height, c)
                    b, *hw, c = h.shape

                    h = self.attention_blocks[block_id](h.reshape(b, -1, c))
                    h = reshape_jax_torch(
                        self.res_conv_blocks[block_id](
                            reshape_jax_torch(h, kernel_dim=1)
                        ),
                        kernel_dim=1,
                    ).reshape(b, *hw, c)
                    h = reshape_jax_torch(
                        h, kernel_dim=self.kernel_dim
                    )  # (bs, c, width, height)

            # Upsampling Block
            h = self.conv_layers[level](h)
            # Shift channels to increase the resolution (only valid for 2D input data)
            h = self.upsample_layers[level](h)

        # Output - Input Residual Connection
        h = self.res_skip_layer(residual=h, skip=skips.pop())
        # Output Layer
        h = self.conv_layers[-1](h)

        return h


class UpsampleFourierGaussian(nn.Module):
    """Performs upsamling on input data using either the Fourier transform
    or gaussian interpolation
    """

    def __init__(
        self,
        new_shape: Sequence[int],
        num_res_blocks: Sequence[int],
        num_channels: Sequence[int],
        num_blocks: int,
        mid_channels: int,
        out_channels: int,
        emb_channels: int,
        kernel_dim: int,
        upsample_ratio: Sequence[int] = None,
        dropout_rate: int = 0.0,
        padding_method: str = "circular",
        use_attention: bool = True,
        num_heads: int = 8,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
        up_method: str = "gaussian",
        normalize_qk: bool = False,
    ):
        super(UpsampleFourierGaussian, self).__init__()

        self.new_shape = new_shape
        self.num_res_blocks = num_res_blocks
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.mid_channels = mid_channels
        self.dropout_rate = dropout_rate
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.kernel_dim = kernel_dim
        self.upsample_ratio = upsample_ratio  # only relevant if up_method == 'fourier'
        self.padding_method = padding_method
        self.use_attention = use_attention
        self.num_heads = num_heads
        self.dtype = dtype
        self.device = device
        self.up_method = up_method

        conv_block_channels = [self.num_channels[0]]

        self.conv_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.res_conv_blocks = nn.ModuleList()

        self.level = 0

        for i in range(self.num_res_blocks[self.level]):
            self.conv_blocks.append(
                ConvBlock(
                    in_channels=conv_block_channels[i],
                    out_channels=self.mid_channels,
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
            self.attention_blocks.append(
                AttentionBlock(
                    in_channels=256,  # TODO: Calculate!
                    num_heads=self.num_heads,
                    normalize_qk=normalize_qk,
                    dtype=self.dtype,
                    device=self.device,
                )
            )

            self.res_conv_blocks.append(
                ResConv1x(
                    in_channels=256,
                    hidden_layer_size=self.mid_channels * 2,
                    out_channels=self.mid_channels,
                    kernel_dim=1,  # due to tokenization
                    dtype=self.dtype,
                    device=self.device,
                )
            )
            conv_block_channels.append(self.mid_channels)

        self.norm = nn.GroupNorm(
            min(max(self.mid_channels // 4, 1), 32),
            self.mid_channels,
            device=self.device,
            dtype=self.dtype,
        )

        self.conv_layers = ConvLayer(
            in_channels=self.mid_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_dim * (3,),
            padding_mode=self.padding_method,
            padding=1,
            case=self.kernel_dim,
            kernel_init=default_init(),
            dtype=self.dtype,
            device=self.device,
        )

        # precomputation required tuples and tensors for fourier based upsampling scheme
        if self.up_method == "fourier":

            # Now calculate the downsampled resolutions
            # calculate list of upsample resolutions
            list_upsample_resolutions = [self.new_shape[1:]]
            for level, channel in enumerate(self.num_channels):
                downsampled_resolution = tuple(
                    [
                        int(res / self.upsample_ratio[level])
                        for res in list_upsample_resolutions[-1]
                    ]
                )
                list_upsample_resolutions.append(downsampled_resolution)
            list_upsample_resolutions = list_upsample_resolutions[::-1]
            list_upsample_resolutions.pop()

            # N ... width, M ... height, W ... depth
            axes_map = {
                1: (2,),  # for input shape (bs, c, N)
                2: (2, 3),  # for input shape (bs, c, M, N)
                3: (2, 3, 4),  # for input shape (bs, c, W, M, N)
            }

            if self.kernel_dim not in axes_map:
                raise ValueError(
                    "Input must be either 2D, 3D or 4D with including the channel dim"
                )

            self.axes = axes_map[self.kernel_dim]
            input_resolution = list_upsample_resolutions[0]

            pad_sizes = [(0, 0)] * (
                self.kernel_dim + 2
            )  # batch and channel dimension added
            for axis in self.axes:
                # self.new_shape is only off shape (c, M, N) for the 2D case thus -1 is necessary
                pad_size = self.new_shape[axis - 1] - input_resolution[axis - 2]
                assert pad_size >= 0, "Padding for Upsampling can't be negative"
                pad_sizes[axis] = (pad_size // 2, pad_size - pad_size // 2)
            pad_sizes = torch.tensor(pad_sizes[::-1], dtype=torch.int).flatten()
            self.pad_sizes = tuple(pad_sizes.tolist())

        elif self.up_method == "gaussian":
            # methods within gaussian are linear and lanczos, where lanczos is only valid for 2D
            self.method = "linear"  # lanczos requires proper device handling!

    def _upsample_fourier_(self, x: Tensor) -> Tensor:
        """Upsampling with the Fourier Transformation for each batch individually"""

        x_fft = torch.fft.fftshift(
            torch.fft.fftn(x, dim=self.axes, norm="forward"), dim=self.axes
        )

        x_fft_padded = F.pad(x_fft, self.pad_sizes, mode="constant")

        x_upsampled = torch.fft.ifftn(
            torch.fft.ifftshift(x_fft_padded, dim=self.axes),
            dim=self.axes,
            norm="forward",
        )

        return torch.real(x_upsampled)

    def _upsample_gaussian_(self, x: Tensor) -> Tensor:
        """Upsampling by using Bilinear or Trilinear Interpolation"""
        # new_shape is (c, w, h) for the 2D case, excluding the batch dimension
        if self.kernel_dim == 1:
            assert len(self.new_shape) == 2, "new_shape needs to be a 2D tuple"
            size = (self.new_shape[1],)
            mode = "linear"
        elif self.kernel_dim == 2:
            assert len(self.new_shape) == 3, "new_shape needs to be a 3D tuple"
            size = (self.new_shape[1], self.new_shape[2])
            mode = "bilinear"
        elif self.kernel_dim == 3:
            assert len(self.new_shape) == 4, "new_shape needs to be a 4D tuple"
            size = (self.new_shape[1], self.new_shape[2], self.new_shape[3])
            mode = "trilinear"
        else:
            raise ValueError("Input must be either 1D, 2D, or 3D without channel")

        if self.method == "linear":
            return F.interpolate(x, size=size, mode=mode, align_corners=True)

        elif self.method == "lanczos":
            assert self.kernel_dim == 2, "LACZOS is only valid for a 2D grid!"
            bs, c, h, w = x.shape
            to_pil = ToPILImage()
            to_tensor = ToTensor()

            resized_img = []
            for i in range(bs):
                img_up = to_tensor(
                    TF.resize(
                        to_pil(x[i]), size=size, interpolation=InterpolationMode.LANCZOS
                    )
                )
                resized_img.append(img_up)
            return torch.stack(resized_img)

        else:
            raise ValueError(
                f"For the upsampling only 'linear' and 'lanczos' interpolations are valid"
            )

    def forward(self, x: Tensor, emb: Tensor) -> Tuple[Tensor, Tensor]:

        h = x

        for block_id in range(self.num_res_blocks[self.level]):
            h = self.conv_blocks[block_id](h, emb)

            if self.use_attention and self.level == 0:
                h = reshape_jax_torch(h, kernel_dim=self.kernel_dim)
                bs, *hw, c = h.shape

                h = self.attention_blocks[block_id](h.reshape(bs, -1, c))

                h = reshape_jax_torch(
                    self.res_conv_blocks[block_id](
                        reshape_jax_torch(h, kernel_dim=1)
                    ).reshape(bs, *hw, c),
                    kernel_dim=self.kernel_dim,
                )

        h = F.silu(self.norm(h))
        h = self.conv_layers(h)

        if self.up_method == "fourier":
            h_up = self._upsample_fourier_(h)
        elif self.up_method == "gaussian":
            h_up = self._upsample_gaussian_(h)
        else:
            raise ValueError(
                "Upsampling method does not exist, choose either 'fourier' or 'gaussian'"
            )

        return h_up, h
