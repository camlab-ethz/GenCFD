# Copyright 2024 The CAM Lab at ETH Zurich.
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

import torch
import torch.nn as nn
import numpy as np
from typing import Sequence
from GenCFD.utils.parser_utils import ArgumentParser

from GenCFD.diffusion.diffusion import NoiseLevelSampling, NoiseLossWeighting

Tensor = torch.Tensor


def permute_tensor(tensor: Tensor, kernel_dim: int) -> Tensor:
    if kernel_dim == 1:
        # Reshape for the 1D case
        return tensor.permute(0, 2, 1)
    elif kernel_dim == 2:
        # Reshape for the 2D case
        return tensor.permute(0, 3, 2, 1)
    elif kernel_dim == 3:
        # Reshape for the 3D case
        return tensor.permute(0, 4, 3, 2, 1)
    else:
        raise ValueError(
            f"Unsupported kernel_dim={kernel_dim}. Only 1D, 2D, and 3D data are valid."
        )


def reshape_jax_torch(tensor: Tensor, kernel_dim: int = None) -> Tensor:
    """
    A jax based dataloader is off shape (bs, width, height, depth, c),
    while a PyTorch based dataloader is off shape (bs, c, depth, height, width).

    It transforms a tensor for the 2D and 3D case as follows:
    - 2D: (bs, c, depth, height, width) <-> (bs, width, height, depth, c)
    - 3D: (bs, c, height, width) <-> (bs, width, height, c)

    Code can be used either dynamics or static.
    - dynamic: if kernel_dim is None
    - static: if kernel_dim
    """
    if kernel_dim is None:
        # Infer kernel_dim dynamically based on tensor.ndim
        kernel_dim = tensor.ndim - 2  # Extract batch_size and channel

    return permute_tensor(tensor, kernel_dim)


def default_init(scale: float = 1e-10):
    """Initialization of weights and biases with scaling"""

    def initializer(tensor: Tensor):
        """We need to differentiate between biases and weights"""

        if tensor.ndim == 1:  # if bias
            bound = torch.sqrt(torch.tensor(3.0)) * scale
            with torch.no_grad():
                return tensor.uniform_(-bound, bound)

        else:  # if weights
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
            std = torch.sqrt(torch.tensor(scale / ((fan_in + fan_out) / 2.0)))
            bound = torch.sqrt(torch.tensor(3.0)) * std  # uniform dist. scaling factor
            with torch.no_grad():
                return tensor.uniform_(-bound, bound)

    return initializer


def get_model_args(
    args: ArgumentParser,
    in_channels: int,
    out_channels: int,
    spatial_resolution: Sequence[int],
    time_cond: bool,
    device: torch.device = None,
    buffer_dict: dict = None,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Return a dictionary of model parameters for the UNet architecture"""

    if args.model_type in ["UNet", "PreconditionedDenoiser"]:
        # General UNet arguments for the 2D case
        args_dict_2d = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "spatial_resolution": spatial_resolution,
            "time_cond": time_cond,
            "resize_to_shape": args.resize_to_shape,
            "use_hr_residual": args.use_hr_residual,
            "num_channels": args.num_channels,
            "downsample_ratio": args.downsample_ratio,
            "num_blocks": args.num_blocks,
            "noise_embed_dim": args.noise_embed_dim,
            "output_proj_channels": 128,
            "input_proj_channels": 128,
            "padding_method": args.padding_method,
            "dropout_rate": args.dropout_rate,
            "use_attention": args.use_attention,
            "use_position_encoding": args.use_position_encoding,
            "num_heads": args.num_heads,
            "normalize_qk": args.normalize_qk,
            "dtype": dtype,
            "device": device,
            "buffer_dict": buffer_dict,
        }
        if args.model_type == "PreconditionedDenoiser":
            args_dict_2d.update({"sigma_data": args.sigma_data})

        args_dict = args_dict_2d

    if args.model_type in ["UNet3D", "PreconditionedDenoiser3D"]:
        # General UNet arguments for the 3D case
        args_dict_3d = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "spatial_resolution": spatial_resolution,
            "time_cond": time_cond,
            "num_channels": (64, 128, 256),
            "downsample_ratio": (2, 2, 2),
            "num_blocks": args.num_blocks,
            "noise_embed_dim": args.noise_embed_dim,
            "input_proj_channels": args.noise_embed_dim,
            "output_proj_channels": args.noise_embed_dim,
            "padding_method": args.padding_method,
            "dropout_rate": args.dropout_rate,
            "use_spatial_attention": (True, True, True),
            "use_position_encoding": args.use_position_encoding,
            "num_heads": args.num_heads,
            "normalize_qk": args.normalize_qk,
            "dtype": dtype,
            "device": device,
            "buffer_dict": buffer_dict,
        }
        if args.model_type == "PreconditionedDenoiser3D":
            args_dict_3d.update({"sigma_data": args.sigma_data})

        args_dict = args_dict_3d

    return args_dict
