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
from unittest import mock
import torch as th
import numpy as np
import diffusion


class DiffusionTest(unittest.TestCase):

  def test_tangent_noise_schedule(self):
    test_cases = [(50.0, 0.0, 1.5), (100.0, -1.5, 1.5)]
    for clip_max, start, end in test_cases:
      with self.subTest(clip_max=clip_max, start=start, end=end):
        sigma = diffusion.tangent_noise_schedule(clip_max, start, end)

        self.assertAlmostEqual(sigma(1.0).numpy(), clip_max, places=3)
        self.assertAlmostEqual(sigma(0.0).numpy(), 0, places=4)

        test_points = th.rand(10) * 0.95 + 0.05
        th.testing.assert_close(
            sigma.inverse(sigma(test_points)), test_points, rtol=1e-5, atol=1e-8
        )

  def test_tangent_schedule_invalid_start_end_points(self):
    test_cases = [(1.0, 0.0), (-2.0, 0.0), (0.0, 2.0)]
    for start, end in test_cases:
      with self.subTest(start=start, end=end):
        with self.assertRaises(ValueError):
          diffusion.tangent_noise_schedule(100.0, start, end)


  def test_power_noise_schedule(self):
    test_cases = [
      {"clip_max": 50.0, "p": 1.0, "start": 0.0, "end": 1.0},
      {"clip_max": 100.0, "p": 2.0, "start": 0.0, "end": 1.0},
      {"clip_max": 100.0, "p": 0.5, "start": 2.0, "end": 100.0},
    ]
    for case in test_cases:
      with self.subTest(clip_max=case["clip_max"], p=case["p"], 
                        start=case["start"], end=case["end"]):
        sigma = diffusion.power_noise_schedule(case["clip_max"], case["p"], 
                                               case["start"], case["end"])
        self.assertAlmostEqual(sigma(1.0), case["clip_max"], places=3)
        self.assertEqual(sigma(0.0), 0)

        test_points = th.rand(10) * 0.95 + 0.05
        th.testing.assert_close(
            sigma.inverse(sigma(test_points)), test_points, rtol=1e-5, atol=1e-8
        )


  def test_power_schedule_invalid_start_end_points(self):
    test_cases = [(0.0, 0.0, 1.0), (1.0, -2.0, 0.0), (1.0, 2.0, 0.0)]
    for p, start, end in test_cases:
      with self.subTest(p=p, start=start, end=end):
        with self.assertRaises(ValueError):
          diffusion.power_noise_schedule(100.0, p, start, end)


  def test_exponential_noise_schedule(self):
    test_cases = [
      {"clip_max": 50.0, "base": 2.0, "start": 0.0, "end": 5.0},
      {"clip_max": 100.0, "base": 3.0, "start": -1.0, "end": 5.0},
    ]
    for case in test_cases:
      with self.subTest(clip_max=case["clip_max"], base=case["base"], 
                        start=case["start"], end=case["end"]):
        sigma = diffusion.exponential_noise_schedule(case["clip_max"], case["base"], 
                                                     case["start"], case["end"])
        self.assertAlmostEqual(sigma(1.0), case["clip_max"], places=3)
        self.assertEqual(sigma(0.0), 0)

        test_points = th.rand(10) * 0.95 + 0.05
        th.testing.assert_close(
            sigma.inverse(sigma(test_points)), test_points, rtol=1e-5, atol=1e-8
        )

  def test_exponential_schedule_invalid_start_end_points(self):
    test_cases = [(1.0, 0.0, 1.0), (2.0, 2.0, 0.0)]
    for base, start, end in test_cases:
      with self.subTest(base=base, start=start, end=end):
        with self.assertRaises(ValueError):
          diffusion.exponential_noise_schedule(100.0, base, start, end)


  def test_logsnr_and_sigma_transforms(self):
    schedules = [diffusion.tangent_noise_schedule, diffusion.power_noise_schedule]
    for schedule in schedules:
      with self.subTest(schedule=schedule):
        sigma = schedule(clip_max=100.0)
        logsnr = diffusion.sigma2logsnr(sigma)
        self.assertTrue(th.isinf(logsnr(th.tensor(0.0))))
        self.assertAlmostEqual(logsnr(1.0), -2 * th.log(th.tensor(100.0)), places=3)

        sigma2 = diffusion.logsnr2sigma(logsnr)
        test_points = th.rand(10) * 0.95 + 0.05
        th.testing.assert_close(
            sigma(test_points), sigma2(test_points), rtol=1e-5, atol=1e-8
        )
        th.testing.assert_close(
            test_points, sigma2.inverse(sigma(test_points)), rtol=1e-5, atol=1e-8
        )


  def test_create_vp(self):
    data_std_vals = [1.0, 2.0]
    for data_std in data_std_vals:
      with self.subTest(data_std=data_std):
        sigma = diffusion.tangent_noise_schedule()
        scheme = diffusion.Diffusion.create_variance_preserving(sigma, data_std)
        test_points = th.rand(10) * 0.99 + 0.01
        # verify that variance is indeed preserved
        variance = th.square(scheme.scale(test_points)) * (
            th.square(th.tensor(data_std)) + th.square(scheme.sigma(test_points))
        )
        th.testing.assert_close(variance, th.full_like(variance, data_std**2), rtol=1e-5, atol=1e-8)
        # verify the inverse is correct
        th.testing.assert_close(
            scheme.sigma.inverse(scheme.sigma(test_points)), test_points, rtol=1e-5, atol=1e-8
        )


  def test_create_ve(self):
    data_std_vals = [1.0, 2.0]
    for data_std in data_std_vals:
      with self.subTest(data_std=data_std):
        sigma = diffusion.tangent_noise_schedule()
        scheme = diffusion.Diffusion.create_variance_exploding(sigma, data_std)
        test_points = th.rand(10) * 0.99 + 0.01
        th.testing.assert_close(scheme.scale(test_points), th.full_like(test_points, 1.0))
        # verify that variance is scaled by data_std
        th.testing.assert_close(
            scheme.sigma(test_points), sigma(test_points) * data_std, rtol=1e-5, atol=1e-8
        )
        # verify the inverse is correct
        th.testing.assert_close(
            scheme.sigma.inverse(scheme.sigma(test_points)), test_points, rtol=1e-5,atol=1e-8
        )


class NoiseLevelSamplingTest(unittest.TestCase):

  def test_uniform_samples(self):
    test_cases = [((4,), True), ((2, 2), False)]
    for sample_shape, uniform_grid in test_cases:
      with self.subTest(sample_shape=sample_shape, uniform_grid=uniform_grid):
        samples = diffusion._uniform_samples(
            th.manual_seed(0), sample_shape, uniform_grid=uniform_grid
        )
        self.assertEqual(samples.shape, sample_shape)
        if uniform_grid:
          self.assertAlmostEqual(th.std(th.diff(th.sort(samples.flatten()).values)).item(), 0.0)

  def test_log_uniform_sampling(self):
    uniform_grid_vals = [True, False]
    sample_shape = (25,)
    clip_min = 0.1
    scheme = mock.Mock(spec=diffusion.Diffusion)
    scheme.sigma_max = 100.0

    for uniform_grid in uniform_grid_vals:
      with self.subTest(uniform_grid=uniform_grid):
        noise_sampling = diffusion.log_uniform_sampling(
            scheme, clip_min, uniform_grid
        )
        samples = noise_sampling(th.manual_seed(1), sample_shape)
        self.assertEqual(samples.shape, sample_shape)
        self.assertGreaterEqual(th.min(samples).item(), clip_min)
        self.assertLessEqual(th.max(samples).item(), scheme.sigma_max)
        if uniform_grid:
          self.assertAlmostEqual(
              th.std(th.diff(th.sort(th.log(samples)).values)).item(), 0.0, places=5
          )


  def test_time_uniform(self):
    uniform_grid_vals = [True, False]
    sample_shape = (25,)
    clip_min = 0.1
    scheme = diffusion.Diffusion.create_variance_exploding(
        diffusion.tangent_noise_schedule()
    )

    for uniform_grid in uniform_grid_vals:
      with self.subTest(uniform_grid=uniform_grid):
        noise_sampling = diffusion.time_uniform_sampling(
            scheme, clip_min, uniform_grid
        )
        samples = noise_sampling(th.manual_seed(0), sample_shape)
        self.assertEqual(samples.shape, sample_shape)
        self.assertGreaterEqual(th.min(samples).item(), clip_min)
        self.assertLessEqual(th.max(samples).item(), scheme.sigma_max)
        if uniform_grid:
          self.assertAlmostEqual(
              th.std(th.diff(th.sort(scheme.sigma.inverse(samples)).values)).item(), 0.0, places=5
          )

  def test_edm_schedule(self):
    sample_shape = (20000,)
    p_mean, p_std = -1.2, 1.2
    scheme = mock.Mock(spec=diffusion.Diffusion)
    scheme.sigma_max = 100.0
    noise_sampling = diffusion.normal_sampling(
        scheme=scheme, p_mean=p_mean, p_std=p_std
    )
    samples = noise_sampling(th.manual_seed(1), sample_shape)
    self.assertEqual(samples.shape, sample_shape)
    self.assertAlmostEqual(th.mean(th.log(samples)).item(), p_mean, places=1)
    self.assertAlmostEqual(th.std(th.log(samples)).item(), p_std, places=1)


class NoiseLossWeightingTest(unittest.TestCase):

  def test_inverse_sigma_squared_schedule(self):
    test_cases = [(4.0, 0.0625), (np.asarray([1.0, 0.5]), np.asarray([1.0, 4]))]
    for sigma, expected_res in test_cases:
      with self.subTest(sigma=sigma, expected_res=expected_res):
        res = diffusion.inverse_squared_weighting(th.as_tensor(sigma))
        self.assertTrue(th.allclose(res, th.as_tensor(expected_res)))


  def test_edm_weighting(self):
    test_cases = [
      (2.0, 4.0, 0.3125),
      (4.0, np.asarray([1.0, 8.0]), np.asarray([1.0625, 0.078125])),
    ]
    for sigma_data, sigma, expected_res in test_cases:
      with self.subTest(sigma_data=sigma_data, sigma=sigma, expected_res=expected_res):
        res = diffusion.edm_weighting(sigma_data)(th.as_tensor(sigma))
        self.assertTrue(th.allclose(res, th.as_tensor(expected_res)))


if __name__ == "__main__":
  unittest.main()
