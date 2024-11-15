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
from diffusion import diffusion, schedulers



class TimeStepSchedulersTest(unittest.TestCase):

    def test_uniform_time(self):
        num_steps = 3
        end_time = 0.2
        expected = [1.0, 0.6, 0.2]
        sigma_schedule = diffusion.tangent_noise_schedule()
        scheme = diffusion.Diffusion.create_variance_exploding(sigma_schedule)

        tspan = schedulers.uniform_time(scheme, num_steps, end_time, end_sigma=None)
        np.testing.assert_allclose(tspan, np.asarray(expected), atol=1e-6)

        tspan = schedulers.uniform_time(scheme, num_steps, end_time=None, end_sigma=sigma_schedule(end_time))
        np.testing.assert_allclose(tspan, np.asarray(expected), atol=1e-6)

    def test_exponential_noise_decay(self):
        num_steps = 4
        start_sigma, end_sigma = 100, 0.1
        expected_noise = np.asarray([100, 10, 1, 0.1])
        sigma_schedule = diffusion.tangent_noise_schedule(clip_max=start_sigma)
        expected_tspan = sigma_schedule.inverse(np.asarray(expected_noise))
        scheme = diffusion.Diffusion.create_variance_exploding(sigma_schedule)
        tspan = schedulers.exponential_noise_decay(scheme, num_steps, end_sigma)
        np.testing.assert_allclose(tspan, np.asarray(expected_tspan), atol=1e-6)

    def test_edm_noise_decay(self):
        num_steps = 3
        start_sigma, end_sigma = 100, 1
        expected_noise = np.asarray([100, 30.25, 1])
        sigma_schedule = diffusion.tangent_noise_schedule(clip_max=start_sigma)
        expected_tspan = sigma_schedule.inverse(expected_noise)
        scheme = diffusion.Diffusion.create_variance_exploding(sigma_schedule)
        tspan = schedulers.edm_noise_decay(scheme, rho=2, num_steps=num_steps, end_sigma=end_sigma)
        np.testing.assert_allclose(tspan, np.asarray(expected_tspan), atol=1e-6)

    

if __name__=="__main__":
    unittest.main()