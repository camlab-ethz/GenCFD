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
from typing import Sequence

from GenCFD.diffusion.diffusion import NoiseLevelSampling, NoiseLossWeighting
from GenCFD.utils.parser_utils import ArgumentParser

# General Denoiser arguments
def get_denoiser_args(
    args: ArgumentParser,
    spatial_resolution: Sequence[int],
    time_cond: bool,
    denoiser: nn.Module,
    noise_sampling: NoiseLevelSampling,
    noise_weighting: NoiseLossWeighting,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Return a dictionary of parameters for the DenoisingModel"""

    denoiser_args = {
        "spatial_resolution": spatial_resolution,
        "time_cond": time_cond,
        "denoiser": denoiser,
        "noise_sampling": noise_sampling,
        "noise_weighting": noise_weighting,
        "num_eval_noise_levels": args.num_eval_noise_levels,
        "num_eval_cases_per_lvl": args.num_eval_cases_per_lvl,
        "min_eval_noise_lvl": args.min_eval_noise_lvl,
        "max_eval_noise_lvl": args.max_eval_noise_lvl,
        "consistent_weight": args.consistent_weight,
        "device": device,
        "dtype": dtype,
        "task": args.task,
    }

    return denoiser_args
