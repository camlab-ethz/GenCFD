# Copyright 2024 The swirl_dynamics Authors.
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
import warnings

Tensor = torch.Tensor
SdeParams = Mapping[str, Any]

class SdeCoefficientFn(Protocol):
    """A callable type for the drift or diffusion coefficients of an SDE."""

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Evaluates the drift or diffusion coefficients."""
        ...

class SdeDynamics(NamedTuple):
    """The drift and diffusion functions that represents the SDE dynamics."""

    drift: SdeCoefficientFn
    diffusion: SdeCoefficientFn

  
class SdeSolver(nn.Module):
    """A callable type implementation a SDE solver.
    
    Attributes:
      terminal_only: If 'True' the solver only returns the terminal state,
        i.e., corresponding to the last time stamp in 'tspan'. If 'False',
        returns the full path containing all steps.
    """

    def __init__(self, rng: torch.Generator, terminal_only: bool = False):
        super().__init__()
        self.terminal_only = terminal_only
        self.rng = rng

    def forward(
            self,
            dynamics: SdeDynamics,
            x0: Tensor,
            tspan: Tensor
      ) -> Tensor:
        """Solves an SDE at given time stamps.

        Args:
          dynamics: The SDE dynamics that evaluates the drift and diffusion
            coefficients.
          x0: Initial condition.
          tspan: The sequence of time points on which the approximate solution
            of the SDE are evaluated. The first entry corresponds to the time for x0.
          rng: A PyTorch generator used to draw realizations of the Wiener processes
            for the SDE.

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

    def __init__(
            self,
            rng: torch.Generator = None,
            time_axis_pos: int = 0, 
            terminal_only: bool = False
    ):
        super().__init__()
        self.rng = rng if rng is not None else torch.Generator().manual_seed(0)
        self.time_axis_pos = time_axis_pos
        self.terminal_only = terminal_only

    def step(
            self,
            dynamics: SdeDynamics,
            x0: Tensor,
            t0: Tensor,
            dt: Tensor
    ) -> Tensor:
        """Advances the current state one step forward in time."""
        raise NotImplementedError

    def forward(
            self,
            dynamics: SdeDynamics,
            x0: Tensor,
            tspan: Tensor
    ) -> Tensor:
        """Solves an SDE by iterating the step function."""

        x_path = [x0]
        current_state = x0
        for i in range(len(tspan) - 1):
            t0 = tspan[i]
            t_next = tspan[i + 1]
            dt = t_next - t0
            current_state = self.step(
                dynamics=dynamics, x0=current_state, t0=t0, dt=dt
                )
            x_path.append(current_state)
        
        out = torch.stack(x_path, dim=0)
        if self.time_axis_pos != 0:
            out = out.movedim(0, self.time_axis_pos)
        return out
    

class EulerMaruyamaStep(nn.Module):
    """The Euler-Maruyama scheme for integrating the Ito SDE"""

    def __init__(self, rng: torch.Generator = None):
        super().__init__()
        self.rng = rng if rng is not None else torch.Generator().manual_seed(0)

    def step(
            self,
            dynamics: SdeDynamics,
            x0: Tensor,
            t0: Tensor,
            dt: Tensor
    ) -> Tensor:
        """Makes one Euler-Maruyama integration step in time."""

        drift_coeffs = dynamics.drift(x0, t0)
        diffusion_coeffs = dynamics.diffusion(x0, t0)

        noise = torch.randn(
            size=x0.shape, generator=self.rng, dtype=x0.dtype, device=x0.device
            )
        return (
            x0 + 
            dt * drift_coeffs + 
            # abs to enable integration backward in time
            diffusion_coeffs * noise * torch.sqrt(torch.abs(dt))
        )
    
class EulerMaruyama(EulerMaruyamaStep, IterativeSdeSolver):
    """Solver using the Euler-Maruyama with iteration (i.e. looping through time steps)."""
    def __init__(self, rng: torch.Generator = None, time_axis_pos: int = 0):
        super().__init__()
        EulerMaruyamaStep.__init__(self, rng=rng)
        IterativeSdeSolver.__init__(self, rng=rng, time_axis_pos=time_axis_pos)


# def EulerMaruyama(time_axis_pos: int | None = None) -> SdeSolver:
#     """Factory function to choose between solvers if different ones are implemented"""
#     time_axis_pos = time_axis_pos or 0
#     return EulerMaruyamaBase(time_axis_pos)
