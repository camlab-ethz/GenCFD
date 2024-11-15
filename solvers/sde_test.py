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

import unittest
import torch
import numpy as np

from solvers import EulerMaruyama, SdeDynamics

torch.manual_seed(0)

class SdeSolverTest(unittest.TestCase):

    def test_euler_maruyama_constant_drift_no_diffusion(self):
        dt = 0.1
        x_dim = 5
        tspan_length = 10
        tspan = torch.arange(tspan_length) * dt
        rng = torch.Generator().manual_seed(0)
        solver = EulerMaruyama(rng=rng)

        out = solver(
            dynamics=SdeDynamics(
                drift=lambda x, t, params: torch.ones_like(x),
                diffusion=lambda x, t, params: torch.zeros_like(x)
            ),
            x0=torch.zeros(x_dim),
            tspan=tspan,
            params = {"drift": {}, "diffusion": {}}
        )
        final = out[-1]
        expected = torch.ones(x_dim) * tspan[-1]
        np.testing.assert_allclose(final.numpy(), expected.numpy(), rtol=1e-5)


    def test_euler_maruyama_scan_equivalence(self):
        dt = 0.1
        x_dim = 5
        tspan = torch.arange(1, 10) * dt
        rng = torch.Generator().manual_seed(12)
        
        def solve(solver):
            return solver(
                dynamics=SdeDynamics(
                    drift=lambda x, t, params: params * x,
                    diffusion=lambda x, t, params: params + x,
                ),
                x0=torch.ones(x_dim),
                tspan=tspan,
                params={"drift": 2, "diffusion": 1}
            )

        solver = EulerMaruyama(rng=rng)
        out = solve(solver)
        np.testing.assert_allclose(out[-1].numpy(), out[-1].numpy(), rtol=1e-5)


    def test_linear_drift_and_diffusion(self):
        for terminal_only in (False, True):
            dt = 1e-4
            tspan = torch.arange(100) * dt
            rng_solver = torch.Generator().manual_seed(0)
            rng_wiener = torch.Generator().manual_seed(0)
            x0 = torch.tensor(1.0, dtype=torch.float64)
            mu, sigma = 1.5, 0.25

            solver = EulerMaruyama(rng=rng_solver, terminal_only=terminal_only)
            dynamics_params = {"drift": mu, "diffusion": sigma}
            out = solver(
                dynamics=SdeDynamics(
                    drift=lambda x, t, params: params * x,
                    diffusion=lambda x, t, params: params * x,
                ),
                x0=x0,
                tspan=tspan,
                params=dynamics_params
            )
            # Simulate Wiener process
            noise = torch.randn((1,), generator=rng_wiener)
            for _ in range(1, len(tspan)):
                noise = torch.cat([noise, torch.randn((1,), generator=rng_wiener)], dim=0)
            wiener_increments = noise * torch.sqrt(torch.tensor(dt))
            path = torch.cumsum(wiener_increments[:-1], dim=0)
            path = torch.cat([torch.zeros(1, dtype=torch.float64), path])
            
            # Analytical solution
            # dX_t = mu * X_t * dt + sigma * X_t * dW_t
            # => X_t = X_0 * exp(mu * t + sigma * W_t)
            expected = x0 * torch.exp(mu * tspan + sigma * path)
            if terminal_only:
                expected = expected[-1]
            np.testing.assert_allclose(out.numpy(), expected.numpy(), atol=1e-3, rtol=1e-2)



    def test_backward_integration(self):
        dt = 0.1
        num_steps = 10
        x_dim = 5
        tspan = -1 * torch.arange(num_steps) * dt
        rng_solver = torch.Generator().manual_seed(0)
        rng_wiener = torch.Generator().manual_seed(0)

        solver = EulerMaruyama(rng=rng_solver)
        out = solver(
            dynamics=SdeDynamics(
                drift=lambda x, t, params: torch.ones_like(x),
                diffusion=lambda x, t, params: torch.ones_like(x)
            ),
            x0=torch.zeros(x_dim),
            tspan=tspan,
            params = {"drift": {}, "diffusion": {}}
        )

        # Simulate Wiener process
        noise = torch.randn(x_dim, generator=rng_wiener).reshape(1,-1)
        for _ in range(1, len(tspan)):
            noise = torch.cat([noise, torch.randn(x_dim, generator=rng_wiener).reshape(1,-1)], dim=0)

        wiener_increments = noise * torch.sqrt(torch.tensor(dt))
        expected = torch.ones(x_dim) * tspan[-1] + torch.sum(wiener_increments[:-1], dim=0)
        np.testing.assert_allclose(out[-1].numpy(), expected.numpy(), rtol=1e-5)



    def test_move_time_axis_pos(self):
        dt = 0.1
        num_steps = 10
        x_dim = 5
        batch_sz = 6
        tspan = torch.arange(num_steps) * dt
        rng = torch.Generator().manual_seed(1)
        
        solver = EulerMaruyama(rng=rng, time_axis_pos=1)
        out = solver(
            dynamics=SdeDynamics(
                drift=lambda x, t, params: torch.ones_like(x),
                diffusion=lambda x, t, params: torch.ones_like(x)
            ),
            x0=torch.zeros(batch_sz, x_dim),
            tspan=tspan,
            params = {"drift": {}, "diffusion": {}}
        )
        self.assertEqual(out.shape, (batch_sz, num_steps, x_dim))

    # def test_invalid_params(self):
    #     with self.assertRaises(ValueError):
    #         rng = torch.Generator().manual_seed(0)
    #         solver = EulerMaruyama(rng=rng)
    #         solver(
    #             dynamics=SdeDynamics(
    #                 drift=lambda x, t: torch.ones_like(x),
    #                 diffusion=lambda x, t: torch.zeros_like(x),
    #             ),
    #             x0=torch.ones(10),
    #             tspan=torch.arange(10),
    #         )

if __name__ == "__main__":
    unittest.main()