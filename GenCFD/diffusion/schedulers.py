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

# ********************
# Time step schedulers
# ********************
import torch
from typing import Protocol
from GenCFD.diffusion import diffusion

Tensor = torch.Tensor


class TimeStepScheduler(Protocol):

    def __call__(self, scheme: diffusion.Diffusion, *args, **kwargs) -> Tensor:
        """Outputs the time steps based on diffusion noise schedule."""
        ...


def uniform_time(
    scheme: diffusion.Diffusion,
    num_steps: int = 256,
    end_time: float | None = 1e-3,
    end_sigma: float | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> Tensor:
    """Time steps uniform in [t_min, t_max]."""

    if (end_time is None and end_sigma is None) or (
        end_time is not None and end_sigma is not None
    ):
        raise ValueError("Exactly one of `end_time` and `end_sigma` must be specified.")

    start = diffusion.MAX_DIFFUSION_TIME
    end = end_time or scheme.sigma.inverse(end_sigma)
    return torch.linspace(start, end, num_steps, dtype=dtype, device=device)


def exponential_noise_decay(
    scheme: diffusion.Diffusion,
    num_steps: int = 256,
    end_sigma: float | None = 1e-3,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> Tensor:
    """Time steps corresponding to exponentially decaying sigma."""

    exponent = torch.arange(num_steps, dtype=dtype, device=device) / (num_steps - 1)
    r = end_sigma / scheme.sigma_max
    sigma_schedule = scheme.sigma_max * torch.pow(r, exponent)
    return scheme.sigma.inverse(sigma_schedule)


def edm_noise_decay(
    scheme: diffusion.Diffusion,
    rho: int = 7,
    num_steps: int = 256,
    end_sigma: float | None = 1e-3,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> Tensor:
    """Time steps corresponding to Eq. 5 in Karras et al."""

    rho_inv = torch.tensor(1.0 / rho)
    sigma_schedule = torch.arange(num_steps, dtype=dtype, device=device) / (
        num_steps - 1
    )
    sigma_schedule *= torch.pow(end_sigma, rho_inv) - torch.pow(
        scheme.sigma_max, rho_inv
    )
    sigma_schedule += torch.pow(scheme.sigma_max, rho_inv)
    sigma_schedule = torch.pow(sigma_schedule, rho)
    return scheme.sigma.inverse(sigma_schedule)
