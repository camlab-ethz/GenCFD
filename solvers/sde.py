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

"""Solvers for stochastic differential equations (SDEs)."""

from collections.abc import Mapping
from typing import Any, NamedTuple, Protocol, Literal, ClassVar
import torch
from torch import nn

Tensor = torch.Tensor
SdeParams = Mapping[str, Any]


class SdeCoefficientFn(Protocol):
    """A callable type for the drift or diffusion coefficients of an SDE."""

    def forward(self, x: Tensor, t: Tensor, params: SdeParams) -> Tensor:
        """Evaluates the drift or diffusion coefficients."""
        ...


class SdeDynamics(NamedTuple):
    """The drift and diffusion functions that represents the SDE dynamics."""

    drift: SdeCoefficientFn
    diffusion: SdeCoefficientFn


def _check_sde_params_fields(params: SdeParams) -> None:
    if not ("drift" in params.keys() and "diffusion" in params.keys()):
        raise ValueError("'params' must contain both 'drift' and 'diffusion' fields.")


def output_drift(
    drift: SdeCoefficientFn,
    x: Tensor,
    t: Tensor,
    params: SdeParams,
    y: Tensor = None,
    lead_time: Tensor = None,
) -> Tensor:
    """Evaluate if y or the lead time is required and output the corresponding
    result
    """
    if y is None and lead_time is None:
        return drift(x=x, t=t, params=params)
    elif y is not None and lead_time is None:
        return drift(x=x, t=t, params=params, y=y)
    elif y is not None and lead_time is not None:
        return drift(x=x, t=t, params=params, y=y, lead_time=lead_time)


class SdeSolver(nn.Module):
    """A callable type implementation a SDE solver.

    Attributes:
      terminal_only: If 'True' the solver only returns the terminal state,
        i.e., corresponding to the last time stamp in 'tspan'. If 'False',
        returns the full path containing all steps.
    """

    def __init__(self, terminal_only: bool = False):
        super().__init__()
        self.terminal_only = terminal_only

    def forward(
        self,
        dynamics: SdeDynamics,
        x0: Tensor,
        tspan: Tensor,
        y: Tensor = None,
        lead_time: Tensor = None,
    ) -> Tensor:
        """Solves an SDE at given time stamps.

        Args:
          dynamics: The SDE dynamics that evaluates the drift and diffusion
            coefficients.
          x0: Initial condition.
          tspan: The sequence of time points on which the approximate solution
            of the SDE are evaluated. The first entry corresponds to the time for x0.

        Returns:
          Integrated SDE trajectory (initial condition included at time position 0).
        """
        raise NotImplementedError


class IterativeSdeSolver(nn.Module):
    """A SDE solver based on an iterative step function using PyTorch

    Attributes:
      time_axis_pos: The index where the time axis should be placed. Defaults
      to the lead axis (index 0).
    """

    def __init__(self, time_axis_pos: int = 0, terminal_only: bool = False):
        super().__init__()
        self.time_axis_pos = time_axis_pos
        self.terminal_only = terminal_only

    def step(
        self,
        dynamics: SdeDynamics,
        x0: Tensor,
        t0: Tensor,
        dt: Tensor,
        params: SdeParams,
        y: Tensor = None,
        lead_time: Tensor = None,
    ) -> Tensor:
        """Advances the current state one step forward in time."""
        raise NotImplementedError

    def forward(
        self,
        dynamics: SdeDynamics,
        x0: Tensor,
        tspan: Tensor,
        params: SdeParams,
        y: Tensor = None,
        lead_time: Tensor = None,
    ) -> Tensor:
        """Solves an SDE by iterating the step function."""

        if not self.terminal_only:
            # store the entire path
            x_path = [x0]

        current_state = x0
        for i in range(len(tspan) - 1):
            t0 = tspan[i]
            t_next = tspan[i + 1]
            dt = t_next - t0
            current_state = self.step(
                dynamics=dynamics,
                x0=current_state,
                t0=t0,
                dt=dt,
                params=params,
                y=y,
                lead_time=lead_time,
            ).detach()  # to avoid memory issues!

            if not self.terminal_only:
                x_path.append(current_state)

        if self.terminal_only:
            return current_state
        else:
            out = torch.stack(x_path, dim=0)
            if self.time_axis_pos != 0:
                out = out.movedim(0, self.time_axis_pos)
            return out


class EulerMaruyamaStep(nn.Module):
    """The Euler-Maruyama scheme for integrating the Ito SDE"""

    def step(
        self,
        dynamics: SdeDynamics,
        x0: Tensor,
        t0: Tensor,
        dt: Tensor,
        params: SdeParams,
        y: Tensor = None,
        lead_time: Tensor = None,
    ) -> Tensor:
        """Makes one Euler-Maruyama integration step in time."""
        _check_sde_params_fields(params)
        # drift_coeffs = dynamics.drift(x0, y, t0, params["drift"], lead_time)
        drift_coeffs = output_drift(
            drift=dynamics.drift,
            x=x0,
            y=y,
            t=t0,
            params=params["drift"],
            lead_time=lead_time,
        )
        diffusion_coeffs = dynamics.diffusion(x0, t0, params["diffusion"])

        noise = torch.randn(size=x0.shape, dtype=x0.dtype, device=x0.device)
        return (
            x0
            + dt * drift_coeffs
            +
            # abs to enable integration backward in time
            diffusion_coeffs * noise * torch.sqrt(torch.abs(dt))
        )


class EulerMaruyama(EulerMaruyamaStep, IterativeSdeSolver):
    """Solver using the Euler-Maruyama with iteration (i.e. looping through time steps)."""

    def __init__(self, time_axis_pos: int = 0, terminal_only: bool = False):
        super().__init__()
        # EulerMaruyamaStep.__init__(self, rng=rng)
        IterativeSdeSolver.__init__(
            self, time_axis_pos=time_axis_pos, terminal_only=terminal_only
        )


# def EulerMaruyama(time_axis_pos: int | None = None) -> SdeSolver:
#     """Factory function to choose between solvers if different ones are implemented"""
#     time_axis_pos = time_axis_pos or 0
#     return EulerMaruyamaBase(time_axis_pos)
