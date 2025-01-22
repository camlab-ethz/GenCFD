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

"""Resizing modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Union

from GenCFD.model.building_blocks.layers.convolutions import ConvLayer

Tensor = torch.Tensor


class FilteredResize(nn.Module):
    """A resizing op followed by a convolution layer.

    Attributes:
      output_size: The target output spatial dimensions for resizing.
      kernel_size: The kernel size of the convolution layer.
      method: The resizing method (passed to `jax.image.resize`).
      padding: The padding type of the convolutions, one of ['SAME', 'CIRCULAR',
        'LATLON', 'LONLAT].
      initializer: The initializer for the convolution kernels.
      use_local: Whether to use unshared weights in the filtering.
      precision: Level of precision used in the convolutional layer.
      dtype: The data type of the input and output.
      params_dtype: The data type of of the weights.
    """

    def __init__(
        self,
        output_size: Sequence[int],
        kernel_size: Union[Sequence[int], int],
        method: str = "bicubic",
        padding_mode: str = "circular",
        initializer: nn.init = nn.init.normal_,
        use_local: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ):
        super(FilteredResize, self).__init__()

        self.output_size = output_size
        self.kernel_size = kernel_size
        self.method = method
        self.padding_mode = padding_mode
        self.initializer = initializer
        self.use_local = use_local
        self.dtype = dtype
        self.device = device

        # compute padding to allow for a padding_mode:
        if isinstance(kernel_size, Sequence):
            self.padding = tuple((k - 1) // 2 for k in self.kernel_size)
        else:
            self.padding = (kernel_size - 1) // 2

        self.conv_layer = None

    def forward(self, inputs: Tensor) -> Tensor:
        """Resizes and filters the input with a convolution layer.

        Args:
          inputs: An input tensor of shape `(*batch_dims, *resized_dims, channels)`,
            where `batch_dims` can be or arbitrary length and `resized_dims` has the
            same length as that of `self.kernel_size`.

        Returns:
          The input resized to target shape.
        """
        if not inputs.ndim > len(self.output_size):
            raise ValueError(
                f"Number of dimensions in x ({inputs.ndim}) must be larger than the"
                f" length of `output_size` ({len(self.output_size)})!"
            )

        if self.padding_mode.lower() not in ["circular", "latlon", "lonlat", "same"]:
            raise ValueError(
                f"Unsupported padding type: {self.padding} - please use one of"
                " ['SAME', 'CIRCULAR', 'LATLON', 'LONLAT']!"
            )

        # Get the output shape:
        batch_ndim = inputs.ndim - len(self.output_size) - 1

        assert batch_ndim > 0, "Interpolating the Batch Size is not possible!"

        output_shape = (*inputs.shape[: batch_ndim + 1], *self.output_size)

        if inputs.ndim > 4:
            self.method = "trilinear"

        resized = F.interpolate(
            inputs, size=output_shape[2:], mode=self.method, align_corners=False
        )

        # We add another convolution layer to undo any aliasing that could have
        # been introduced by the resizing step.

        if self.conv_layer is None:
            self.conv_layer = ConvLayer(
                in_channels=inputs.shape[1],
                out_channels=inputs.shape[1],
                kernel_size=self.kernel_size,
                padding_mode=self.padding_mode,
                padding=self.padding,
                use_local=self.use_local,
                case=len(inputs.shape) - 2,
                dtype=self.dtype,
                device=self.device,
            )
            # initialize convolution weights
            self.initializer(self.conv_layer.weight)

        out = self.conv_layer(resized)

        return out


class MergeChannelCond(nn.Module):
    """Base class for merging conditional inputs along the channel dimension."""

    def __init__(
        self, embed_dim, kernel_size, resize_method="cubic", padding_mode="circular"
    ):
        super(MergeChannelCond, self).__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.resize_method = resize_method
        self.padding_mode = padding_mode


class InterpConvMerge(MergeChannelCond):
    """Merges conditional inputs through interpolation and convolutions."""

    def __init__(
        self,
        embed_dim,
        kernel_size,
        resize_method="cubic",
        padding_mode="circular",
        dtype=torch.float32,
        device: torch.device = None,
    ):
        super(InterpConvMerge, self).__init__(
            embed_dim, kernel_size, resize_method, padding_mode
        )
        self.dtype = dtype
        self.device = device

        self.conv_layer = None
        self.resize_layer = None
        self.layer_norm = None

        if isinstance(kernel_size, Sequence):
            self.padding = tuple((k - 1) // 2 for k in self.kernel_size)
        else:
            self.padding = (kernel_size - 1) // 2

        self.silu = F.silu()

    def forward(self, x: Tensor, cond: dict[str, Tensor] = None):
        """
        Merges conditional inputs along the channel dimension.

        Args:
          x: The main model input.
          cond: A dictionary of conditional inputs. Those with keys that start with
            "channel:" are processed here while all others are omitted.

        Returns:
          Model input merged with channel conditions.
        """

        out_spatial_shape = x.shape[-3:-1]  # The target spatial shape (D, H, W)

        for key, value in sorted(cond.items()):
            if key.startswith("channel:"):
                if value.ndim != x.ndim:
                    raise ValueError(
                        f"Channel condition `{key}` does not have the same ndim"
                        f" ({value.ndim}) as x ({x.ndim})!"
                    )

                if value.shape[-3:-1] != out_spatial_shape:
                    if self.resize_layer is None:
                        self.resize_layer = FilteredResize(
                            output_size=x.shape[2:],
                            kernel_size=self.kernel_size,
                            method=self.resize_method,
                            padding_mode=self.padding_mode,
                            dtype=self.dtype,
                            device=self.device,
                        )
                    value = self.resize_layer(value)

                    if self.layer_norm is None:
                        # Layer norm uses (C, H, W) and excludes the Batch Size
                        self.layer_norm = nn.LayerNorm(
                            value.shape[1:], device=self.device, dtype=self.dtype
                        )

                    value = self.layer_norm(value)
                    value = self.silu(value)

                    if self.conv_layer is None:
                        self.conv_layer = ConvLayer(
                            in_channels=value.shape[1],
                            out_channels=self.embed_dim,
                            kernel_size=self.kernel_size,
                            padding_mode=self.padding_mode,
                            padding=self.padding,
                            dtype=self.dtype,
                            device=self.device,
                        )

                    value = self.conv_layer(value)

                x = torch.cat([x, value], dim=1)

        return x
