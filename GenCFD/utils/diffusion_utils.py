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

from argparse import ArgumentParser
import torch
import GenCFD.diffusion.diffusion as dfn_lib
from GenCFD.diffusion.samplers import DenoiseFn
from GenCFD.diffusion.schedulers import TimeStepScheduler
import GenCFD.diffusion.schedulers as schedulers
from GenCFD.solvers.sde import SdeSolver


def get_diffusion_scheme(
    args: ArgumentParser, device: torch.device = None
) -> dfn_lib.Diffusion:
    """Create the diffusion scheme"""

    try:
        diffusion_scheme_fn = getattr(dfn_lib.Diffusion, args.diffusion_scheme)
    except AttributeError:
        raise ValueError(f"Invalid diffusion scheme: {args.diffusion_scheme}")

    try:
        sigma_fn = getattr(dfn_lib, args.sigma)
    except AttributeError:
        raise ValueError(f"Invalid sigma function: {args.sigma}")

    diffusion_scheme = diffusion_scheme_fn(
        sigma=sigma_fn(device=device),
        data_std=args.sigma_data,
    )

    return diffusion_scheme


def get_noise_sampling(
    args: ArgumentParser, device: torch.device = None
) -> dfn_lib.NoiseLevelSampling:
    """Create a noise sampler"""

    diffusion_scheme = get_diffusion_scheme(args, device)
    try:
        noise_sampling_fn = getattr(dfn_lib, args.noise_sampling)
    except AttributeError:
        raise ValueError(f"Invalid noise sampling scheme: {args.noise_sampling}")

    noise_sampling = noise_sampling_fn(
        diffusion_scheme, clip_min=1e-4, uniform_grid=True, device=device
    )

    return noise_sampling


def get_noise_weighting(
    args: ArgumentParser, device: torch.device = None
) -> dfn_lib.NoiseLossWeighting:
    """Create a noise weighting scheme"""

    try:
        noise_weighting_fn = getattr(dfn_lib, args.noise_weighting)
    except AttributeError:
        raise ValueError(f"Invalid noise weighting scheme: {args.noise_weighting}")

    noise_weighting = noise_weighting_fn(data_std=args.sigma_data, device=device)

    return noise_weighting


def get_time_step_scheduler(
    args: ArgumentParser,
    scheme: dfn_lib.Diffusion,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> TimeStepScheduler:
    """Get time step scheduler"""

    try:
        tspan_fn = getattr(schedulers, args.time_step_scheduler)
    except AttributeError:
        raise ValueError(f"Invalid time step scheduler: {args.time_step_scheduler}")

    tspan = tspan_fn(
        scheme=scheme,
        rho=args.rho,
        num_steps=args.sampling_steps,
        end_sigma=1e-3,
        dtype=dtype,
        device=device,
    )

    return tspan


def get_sampler_args(
    args: ArgumentParser,
    input_shape: tuple[int, ...],
    scheme: dfn_lib.Diffusion,
    denoise_fn: DenoiseFn,
    tspan: TimeStepScheduler,
    integrator: SdeSolver,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> dict:

    return {
        "input_shape": input_shape,
        "scheme": scheme,
        "denoise_fn": denoise_fn,
        "tspan": tspan,
        "integrator": integrator,
        "guidance_transforms": (),
        "apply_denoise_at_end": args.apply_denoise_at_end,
        "return_full_paths": args.return_full_paths,
        "device": device,
        "dtype": dtype,
    }
