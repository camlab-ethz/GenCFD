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

"""Diffusion samplers."""

# from collections.abc import Mapping, Sequence
from typing import Any, Protocol, Sequence, Mapping

import torch
from torch.autograd import grad
import numpy as np

from GenCFD.diffusion import diffusion, guidance
from GenCFD.solvers import sde


Tensor = torch.Tensor
TensorMapping = Mapping[str, Tensor]
Params = Mapping[str, Any]


class DenoiseFn(Protocol):

    def __call__(
        self, x: Tensor, sigma: Tensor, cond: TensorMapping | None
    ) -> Tensor: ...


ScoreFn = DenoiseFn


def dlog_dt(f: diffusion.ScheduleFn) -> diffusion.ScheduleFn:
    """Returns d/dt log(f(t)) = ḟ(t)/f(t) given f(t)."""
    # def func(t: Tensor) -> Tensor:
    #   if not t.requires_grad:
    #     t = t.requires_grad_(True)
    #     out = f(t)
    #     if not out.requires_grad and torch.all(out == 1):
    #       return out.requires_grad_(True)
    #   else:
    #     return out
    return lambda t: grad(torch.log(f(t)), t, create_graph=True)[0]


def dsquare_dt(f: diffusion.ScheduleFn) -> diffusion.ScheduleFn:
    """Returns d/dt (f(t))^2 = 2ḟ(t)f(t) given f(t)."""
    return lambda t: grad(torch.square(f(t)), t, create_graph=True)[0]


def denoiser2score(denoise_fn: DenoiseFn, scheme: diffusion.Diffusion) -> ScoreFn:
    """Converts a denoiser to the corresponding score function."""

    def _score(x: Tensor, sigma: Tensor, cond: TensorMapping | None = None) -> Tensor:
        # Reference: eq. 74 in Karras et al. (https://arxiv.org/abs/2206.00364).
        scale = scheme.scale(scheme.sigma.inverse(sigma))
        x_hat = x / scale
        target = denoise_fn(x_hat, sigma, cond)
        return (target - x_hat) / (scale * sigma**2)

    return _score


def denoise_fn_output(
    denoise_fn: DenoiseFn,
    x: Tensor,
    sigma: Tensor,
    cond: TensorMapping | None = None,
    y: Tensor = None,
    lead_time: Tensor = None,
) -> Tensor:
    """Depending on the task 'y' and the 'lead_time' compute the result of the
    denoise_fn
    """
    if y is None and lead_time is None:
        return denoise_fn(x, sigma, cond)

    elif y is not None and lead_time is None:
        return denoise_fn(x, sigma, y, cond)

    elif y is not None and lead_time is not None:
        return denoise_fn(x, sigma, y, lead_time, cond)


# ********************
# Samplers
# ********************


class Sampler:
    """Base class for denoising-based diffusion samplers.

    Attributes:
      input_shape: The tensor shape of a sample (excluding any batch dimensions).
      scheme: The diffusion scheme which contains the scale and noise schedules.
      denoise_fn: A function to remove noise from input data. Must handle batched
        inputs, noise levels and conditions.
      tspan: Full diffusion time steps for iterative denoising, decreasing from 1
        to (approximately) 0.
      guidance_transforms: An optional sequence of guidance transforms that
        modifies the denoising function in a post-process fashion.
      apply_denoise_at_end: If `True`, applies the denoise function another time
        to the terminal states, which are typically at a small but non-zero noise
        level.
      return_full_paths: If `True`, the output of `.generate()` and `.denoise()`
        will contain the complete sampling paths. Otherwise only the terminal
        states are returned.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        scheme: diffusion.Diffusion,
        denoise_fn: DenoiseFn,
        tspan: Tensor,
        guidance_transforms: Sequence[guidance.Transform] = (),
        apply_denoise_at_end: bool = True,
        return_full_paths: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.input_shape = input_shape
        self.scheme = scheme
        self.denoise_fn = denoise_fn
        self.tspan = tspan
        self.guidance_transforms = guidance_transforms
        self.apply_denoise_at_end = apply_denoise_at_end
        self.return_full_paths = return_full_paths
        self.device = device
        self.dtype = dtype

    def generate(
        self,
        num_samples: int,
        y: Tensor = None,
        lead_time: Tensor = None,
        cond: TensorMapping | None = None,
        guidance_inputs: TensorMapping | None = None,
    ) -> Tensor:
        """Generates a batch of diffusion samples from scratch.

        Args:
          num_samples: The number of samples to generate in a single batch.
          cond: Explicit conditioning inputs for the denoising function. These
            should be provided **without** batch dimensions (one should be added
            inside this function based on `num_samples`).
          y: is the output and result of the solver
          lead_time: keeps track not of the diffusion time but of the timestep of the solver
            this is relevant for an all to all training strategy
          guidance_inputs: Inputs used to construct the guided denoising function.
            These also should in principle not include a batch dimension.

        Returns:
          The generated samples.
        """
        if self.tspan is None or self.tspan.ndim != 1:
            raise ValueError("`tspan` must be a 1-d Tensor.")

        x_shape = (num_samples,) + self.input_shape
        x1 = torch.randn(x_shape, dtype=self.dtype, device=self.device)
        x1 = x1 * self.scheme.sigma(self.tspan[0]) * self.scheme.scale(self.tspan[0])

        if cond is not None:
            cond = {
                k: v.repeat(num_samples, *([1] * (v.dim() - 1)))
                for k, v in cond.items()
            }

        denoised = self.denoise(
            noisy=x1,
            tspan=self.tspan,
            y=y,
            lead_time=lead_time,
            cond=cond,
            guidance_inputs=guidance_inputs,
        )

        samples = denoised[-1] if self.return_full_paths else denoised
        if self.apply_denoise_at_end:
            denoise_fn = self.get_guided_denoise_fn(guidance_inputs=guidance_inputs)
            samples = denoise_fn_output(
                denoise_fn=denoise_fn,
                x=samples / self.scheme.scale(self.tspan[-1]),
                sigma=self.scheme.sigma(self.tspan[-1]),
                cond=cond,
                y=y,
                lead_time=lead_time,
            )

            if self.return_full_paths:
                denoised = torch.cat([denoised, samples.unsqueeze(0)], axis=0)

        return denoised if self.return_full_paths else samples

    def denoise(
        self,
        noisy: Tensor,
        tspan: Tensor,
        y: Tensor = None,
        lead_time: Tensor = None,
        cond: TensorMapping | None = None,
        guidance_inputs: TensorMapping | None = None,
    ) -> Tensor:
        """Applies iterative denoising to given noisy states.

        Args:
          noisy: A batch of noisy states (all at the same noise level). Can be fully
            noisy or partially denoised.
          tspan: A decreasing sequence of diffusion time steps within the interval
            [1, 0). The first element aligns with the time step of the `noisy`
            input.
          cond: (Optional) Conditioning inputs for the denoise function. The batch
            dimension should match that of `noisy`.
          guidance_inputs: Inputs for constructing the guided denoising function.

        Returns:
          The denoised output.
        """
        raise NotImplementedError

    def get_guided_denoise_fn(self, guidance_inputs: Mapping[str, Tensor]) -> DenoiseFn:
        """Returns a guided denoise function."""
        denoise_fn = self.denoise_fn
        for transform in self.guidance_transforms:
            denoise_fn = transform(denoise_fn, guidance_inputs)
        return denoise_fn


class SdeSampler(Sampler):
    """Draws samples by solving an SDE.

    Attributes:
      integrator: The SDE solver for solving the sampling SDE.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        scheme: diffusion.Diffusion,
        denoise_fn: DenoiseFn,
        tspan: Tensor,
        integrator: sde.SdeSolver = None,
        guidance_transforms: Sequence[guidance.Transform] = (),
        apply_denoise_at_end: bool = True,
        return_full_paths: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            input_shape=input_shape,
            scheme=scheme,
            denoise_fn=denoise_fn,
            tspan=tspan,
            guidance_transforms=guidance_transforms,
            apply_denoise_at_end=apply_denoise_at_end,
            return_full_paths=return_full_paths,
            device=device,
            dtype=dtype,
        )
        self.integrator = integrator

    def denoise(
        self,
        noisy: Tensor,
        tspan: Tensor,
        y: Tensor = None,
        lead_time: Tensor = None,
        cond: TensorMapping | None = None,
        guidance_inputs: TensorMapping | None = None,
    ) -> Tensor:
        """Applies iterative denoising to given noisy states."""
        if self.integrator is None:
            self.integrator = sde.EulerMaruyama(terminal_only=True)

        if self.integrator.terminal_only and self.return_full_paths:
            raise ValueError(
                f"Integrator type `{type(self.integrator)}` does not support"
                " returning full paths."
            )

        params = dict(
            drift=dict(guidance_inputs=guidance_inputs, cond=cond), diffusion={}
        )

        denoised = self.integrator(
            dynamics=self.dynamics,
            x0=noisy,
            tspan=tspan,
            params=params,
            y=y,
            lead_time=lead_time,
        )
        # SDE solvers may return either the full paths or the terminal state only.
        # If the former, the lead axis should be time.
        samples = denoised if self.integrator.terminal_only else denoised[-1]
        return denoised if self.return_full_paths else samples

    @property
    def dynamics(self) -> sde.SdeDynamics:
        """Drift and diffusion terms of the sampling SDE.

        In score function:

          dx = [ṡ(t)/s(t) x - 2 s(t)²σ̇(t)σ(t) ∇pₜ(x)] dt + s(t) √[2σ̇(t)σ(t)] dωₜ,

        obtained by substituting eq. 28, 34 of Karras et al.
        (https://arxiv.org/abs/2206.00364) into the reverse SDE formula - eq. 6 in
        Song et al. (https://arxiv.org/abs/2011.13456). Alternatively, it may be
        rewritten in terms of the denoise function (plugging in eq. 74 of
        Karras et al.) as:

          dx = [2 σ̇(t)/σ(t) + ṡ(t)/s(t)] x - [2 s(t)σ̇(t)/σ(t)] D(x/s(t), σ(t)) dt
            + s(t) √[2σ̇(t)σ(t)] dωₜ

        where s(t), σ(t) are the scale and noise schedule of the diffusion scheme
        respectively.
        """

        def _drift(
            x: Tensor,
            t: Tensor,
            params: Params,
            y: Tensor = None,
            lead_time: Tensor = None,
        ) -> Tensor:
            assert t.ndim == 0, "`t` must be a scalar."
            denoise_fn = self.get_guided_denoise_fn(
                guidance_inputs=params["guidance_inputs"]
            )
            s, sigma = self.scheme.scale(t), self.scheme.sigma(t)
            x_hat = x / s
            if not t.requires_grad:
                t.requires_grad_(True)
            dlog_sigma_dt = dlog_dt(self.scheme.sigma)(t)
            dlog_s_dt = dlog_dt(self.scheme.scale)(t)
            drift = (2 * dlog_sigma_dt + dlog_s_dt) * x
            denoiser_output = denoise_fn_output(
                denoise_fn=denoise_fn,
                x=x_hat,
                sigma=sigma,
                cond=params["cond"],
                y=y,
                lead_time=lead_time,
            )
            # denoise_fn(x_hat, sigma, params["cond"])
            drift = drift - 2 * dlog_sigma_dt * s * denoiser_output
            return drift

        def _diffusion(x: Tensor, t: Tensor, params: Params) -> Tensor:
            del x, params
            assert t.ndim == 0, "`t` must be a scalar."
            if not t.requires_grad:
                t.requires_grad_(True)
            dsquare_sigma_dt = dsquare_dt(self.scheme.sigma)(t)
            return torch.sqrt(dsquare_sigma_dt) * self.scheme.scale(t)

        return sde.SdeDynamics(_drift, _diffusion)
