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

"""Solvers for deterministic ordinary differential equations (ODEs)."""

import torch
import torch.nn as nn
import numpy as np

from typing import Any, Protocol, Optional
from collections.abc import Mapping
from dataclasses import dataclass

Tensor = torch.Tensor
OdeParams = Mapping[str, Any]


class OdeDynamics(nn.Module):
    """Dynamics interface."""

    def forward(self, x: Tensor, t: Tensor, params: OdeParams) -> Tensor:
        """Evaluate the instantaneous dynamics."""
        ...


def nn_module_to_dynamics(
    module: nn.Module, autonomous: bool = True, **static_kwargs
) -> OdeDynamics:
    """Generates an `OdeDynamics` callable from a torch.nn module.
    Args: 
        module: neural network architecture used for predicting the score
        autonomous: time dependent system if True else non autonomous system
    
    Returns:
        OdeDynamics: Forward call of the neural network module to get the predicted score.
    """

    def _dynamics_func(
        x: Tensor, 
        t: Tensor, 
        params: OdeParams, 
        y: Optional[Tensor] = None, 
        lead_time: Optional[Tensor] = None
    ) -> Tensor:
        """OdeParams can be used for additional ode solver specific parameters"""

        args = (x,) if autonomous else (x, t)

        return module.forward(*args, **static_kwargs)

    return _dynamics_func


class OdeSolver(nn.Module):
    """Solver interface."""

    def forward(
        self, 
        func: OdeDynamics, 
        x0: Tensor, 
        tspan: Tensor, 
        params: OdeParams,
        y: Optional[Tensor] = None,
        lead_time: Optional[Tensor] = None
    ) -> Tensor:
        """Solves an ordinary differential equation."""
        ...


@dataclass
class IterativeOdeSolver(nn.Module):
    """ODE solver based on on an iterative step function.

    Attributes:
        time_axis_pos: move the time axis to the specified position in the output
        tensor (by default it is at the 0th position).
        terminal_only: pass only the clean last sample instead of the complete path.
    """


    time_axis_pos: int = 0 
    terminal_only: bool = False

    def __post_init__(self):
        super().__init__()

    def step(
        self, 
        func: OdeDynamics, 
        x0: Tensor, 
        t0: Tensor, 
        dt: Tensor, 
        params: OdeParams,
        y: Optional[Tensor] = None,
        lead_time: Optional[Tensor] = None
    ) -> Tensor:
        """Advances the current state one step forward in time."""
        raise NotImplementedError("IterativeOdeSolver must implement `step` method.")


    def forward(
        self, 
        func: OdeDynamics, 
        x0: Tensor, 
        tspan: Tensor, 
        params: OdeParams,
        y: Optional[Tensor] = None,
        lead_time: Optional[Tensor] = None
    ) -> Tensor:
        """Solves an ODE at given time stamps."""

        if not self.terminal_only:
            # store the entire path
            x_path = [x0]

        current_state = x0
        for i in range(len(tspan) - 1):
            t0 = tspan[i]
            t_next = tspan[i + 1]
            dt = t_next - t0
            current_state = self.step(
                func=func,
                x0=current_state,
                t0=t0,
                dt=dt,
                params=params,
                y=y,
                lead_time=lead_time,
            ).detach() # to avoid unnecessary memory allocation

            if not self.terminal_only:
                x_path.append(current_state)

        if self.terminal_only:
            return current_state
        else:
            out = torch.stack(x_path, dim=0)
            if self.time_axis_pos != 0:
                out = out.movedim(0, self.time_axis_pos)
            return out


@dataclass
class ExplicitEuler(IterativeOdeSolver):
    """1st order Explicit Euler scheme."""

    def step(
        self, 
        func: OdeDynamics, 
        x0: Tensor, 
        t0: Tensor, 
        dt: Tensor, 
        params: OdeParams,
        y: Optional[Tensor] = None,
        lead_time: Optional[Tensor] = None
    ) -> Tensor:
        return x0 + dt * func(x0, t0, params, y, lead_time)


@dataclass
class HeunsMethod(IterativeOdeSolver):
    """2nd order Heun's method."""

    def step(
        self, 
        func: OdeDynamics, 
        x0: Tensor, 
        t0: Tensor, 
        dt: Tensor, 
        params: OdeParams,
        y: Optional[Tensor] = None,
        lead_time: Optional[Tensor] = None
    ) -> Tensor:

        k1 = func(x0, t0, params, y, lead_time)
        k2 = func(x0 + dt * k1, t0 + dt, params, y, lead_time)
        return x0 + dt * (k1 + k2) / 2


@dataclass
class RungeKutta4(IterativeOdeSolver):
    """4th order Runge-Kutta scheme."""

    def step(
        self, 
        func: OdeDynamics, 
        x0: Tensor, 
        t0: Tensor, 
        dt: Tensor, 
        params: OdeParams,
        y: Optional[Tensor] = None,
        lead_time: Optional[Tensor] = None
    ) -> Tensor:
        k1 = func(x0, t0, params, y, lead_time)
        k2 = func(x0 + dt * k1 / 2, t0 + dt / 2, params, y, lead_time)
        k3 = func(x0 + dt * k2 / 2, t0 + dt / 2, params, y, lead_time)
        k4 = func(x0 + dt * k3, t0 + dt, params, y, lead_time)
        return x0 + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


@dataclass
class OneStepDirect(IterativeOdeSolver):
    """Solver that directly returns function output as next time step."""

    def step(
        self, 
        func: OdeDynamics, 
        x0: Tensor, 
        t0: Tensor, 
        dt: Tensor, 
        params: OdeParams,
        y: Optional[Tensor] = None,
        lead_time: Optional[Tensor] = None
    ) -> Tensor:
        """Performs a single prediction step: `x_{n+1} = f(x_n, t_n)`."""
        return func(x0, t0, params, y, lead_time)
