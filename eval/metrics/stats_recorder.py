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
from typing import Sequence

Tensor = torch.Tensor

class StatsRecorder:
    """StatsRecorder which keeps track of metrics for the Ground Truth 
    and the Generated Data
    """
    def __init__(self, 
                 batch_size: int, 
                 ndim: int, 
                 channels: int, 
                 data_shape: Sequence[int],
                 monte_carlo_samples: int,
                 num_samples: int = 1000,
                 device: torch.device = None
        ):
        self.device = device

        # information about the dataset
        self.batch_size = batch_size
        self.ndim = ndim # 3D or 2D dataset
        self.axis = self._set_axis(ndim) # axis for channel wise mean
        self.channels = channels
        self.data_shape = data_shape
        self.monte_carlo_samples = monte_carlo_samples
        self.num_samples = num_samples

        # storage for sampled points
        self.gen_samples = torch.zeros((monte_carlo_samples, num_samples, channels), device=device)
        self.gt_samples = torch.zeros((monte_carlo_samples, num_samples, channels), device=device)

        # Pre-sample indices for unique points
        self.sample_indices = self._sample_unique_indices(data_shape[1:], ndim) # only spatial resolution

        # initialize mean and std
        self.mean_gt = torch.zeros(data_shape, device=device)
        self.mean_gen = torch.zeros(data_shape, device=device)
        self.std_gt = torch.zeros(data_shape, device=device)
        self.std_gen = torch.zeros(data_shape, device=device)

        # track the number of observation
        self.observation = 0


    def _set_axis(self, ndim: int):
        """Sets the axis for mean and std deviation based on ndim."""

        if ndim == 2:
            return (1, 2)
        elif ndim == 3:
            return (1, 2, 3)
        else:
            raise ValueError(f"Only 2D or 3D data supported, got {ndim}D data.")
    

    def _sample_unique_indices(self, spatial_resolution: Sequence[int], ndim: int):
        """Sample unique pixel values to store the generated and gt results"""

        if ndim == 2:
            height, width = spatial_resolution
            # random unique indices
            flat_indices = torch.randperm(width * height)[:self.num_samples]
            y_indices = flat_indices // width
            x_indices = flat_indices % width
            return torch.stack((x_indices, y_indices), dim=-1).to(self.device) 

        elif ndim == 3:
            depth, height, width = spatial_resolution
            # random unique indices
            flat_indices = torch.randperm(width * height * depth)[:self.num_samples]
            z_indices = flat_indices // (width * height)
            y_indices = (flat_indices % (width * height)) // width
            x_indices = flat_indices % width
            return torch.stack((x_indices, y_indices, z_indices), dim=-1).to(self.device) 

        else:
            raise ValueError(f"Only 2D or 3D supported, not {ndim}D")

    
    def _validate_data(self, gen_data: Tensor, gt_data: Tensor):
        """Validates the dimensionality and types of gen_data and gt_data."""

        if not (isinstance(gen_data, Tensor) and isinstance(gt_data, Tensor)):
            raise TypeError("gen_data and gt_data must be PyTorch Tensors.")
        
        expected_dim = 4 if self.ndim == 2 else 5
        assert gen_data.ndim == expected_dim and gt_data.ndim == expected_dim, (
            f"Expected shape (bs, c, x, y) for 2D or (bs, c, x, y, z) for 3D,"
            f"got gen_data {gen_data.shape} and gt_data {gt_data.shape}."
        )


    def update_step(self, gen_data: Tensor, gt_data: Tensor):
        """Updates all relevant metrics"""

        self._validate_data(gen_data, gt_data)
        self.update_step_mean_and_std(gen_data, gt_data)
        self.sample_points(gen_data, gt_data, self.ndim)

        self.observation += self.batch_size

    
    def sample_points(self, gen_data: Tensor, gt_data: Tensor, ndim: int):
        """Extract sampled points based on precomputed indices."""

        # Retrieve the current step
        current_step = self.observation # current number of observations
        final_step = self.observation + self.batch_size

        x, y = self.sample_indices[:, 0], self.sample_indices[:, 1]

        if ndim == 2:
            sampled_gen = gen_data[..., y, x]
            sampled_gt = gt_data[..., y, x]
        elif ndim == 3:
            z = self.sample_indices[:, 2]
            sampled_gen = gen_data[..., z, y, x]
            sampled_gt = gt_data[..., z, y, x]
        else:
            raise ValueError(f"Only 2D or 3D supported, not {self.ndim}D")

        # Reshape to (batch_size, num_samples, channels)
        sampled_gen = sampled_gen.permute(0, 2, 1)
        sampled_gt = sampled_gt.permute(0, 2, 1)

        # Store the current Monte Carlo step
        self.gen_samples[current_step : final_step] = sampled_gen
        self.gt_samples[current_step : final_step] = sampled_gt


    def update_step_mean_and_std(self, gen_data: Tensor, gt_data: Tensor):
        """Keeps track of the mean of the dataset
        
        Instead of computing the mean of the error, the ground truth
        and the generated results, this method updates the mean of the 
        dataset itself for the gt as well as for the generated data
        """

        m = self.observation # current number of observations
        n = self.batch_size # observations added
        # new number of observations would then be m + n
        mean_gt_data = gt_data.mean(dim=0)
        mean_gen_data = gen_data.mean(dim=0)
        std_gt_data = gt_data.std(dim=0) if gt_data.size(0) > 1 else gt_data.std(dim=0, correction=0)
        std_gen_data = gen_data.std(dim=0) if gen_data.size(0) > 1 else gen_data.std(dim=0, correction=0)

        # first we will update the standard deviation before we update the mean
        var_gt = (
            (m / (m + n)) * (self.std_gt ** 2) + 
            (n / (m + n)) * (std_gt_data ** 2) + 
            (m * n / (m + n) ** 2) * (self.mean_gt - mean_gt_data) ** 2
        )
        self.std_gt = torch.sqrt(var_gt)

        var_gen = (
            (m / (m + n)) * (self.std_gen ** 2) + 
            (n / (m + n)) * (std_gen_data ** 2) + 
            (m * n / (m + n) ** 2) * (self.mean_gen - mean_gen_data) ** 2
        )
        self.std_gen = torch.sqrt(var_gen)

        # update the mean values
        self.mean_gt = m / (m + n) * self.mean_gt + n / (m + n) * mean_gt_data
        self.mean_gen = m / (m + n) * self.mean_gen + n / (m + n) * mean_gen_data