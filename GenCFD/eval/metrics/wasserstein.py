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

import torch
import numpy as np
from scipy.stats import wasserstein_distance
from typing import Union

Tensor = torch.Tensor


def wasserstein_distance_1d(u_values: Tensor, v_values: Tensor, p: int = 1):
    """
    Compute the 1D Wasserstein distance (or general p-Wasserstein distance)
    between two distributions.

    p is a positive parameter; p = 1 gives the Wasserstein distance, p = 2
    gives the energy distance.

    References
    ----------
    .. [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
           Munos "The Cramer Distance as a Solution to Biased Wasserstein
           Gradients" (2017). :arXiv:`1705.10743`.

    Parameters
    ----------
    u_values, v_values : array_like
        Values observed in the (empirical) distribution.
    u_weights, v_weights : array_like, optional
        Weight for each value. If unspecified, each value is assigned the same
        weight.
        `u_weights` (resp. `v_weights`) must have the same length as
        `u_values` (resp. `v_values`). If the weight sum differs from 1, it
        must still be positive and finite so that the weights can be normalized
        to sum to 1.

    Returns
    -------
    distance : float
        The computed distance between the distributions.
    """

    u_sorter = torch.argsort(u_values)
    v_sorter = torch.argsort(v_values)

    all_values = torch.cat((u_values, v_values))
    all_values = torch.sort(all_values).values

    # Compute the differences between pairs of successive values of u and v.
    deltas = torch.diff(all_values)

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = torch.searchsorted(u_values[u_sorter], all_values[:-1], right=True)
    v_cdf_indices = torch.searchsorted(v_values[v_sorter], all_values[:-1], right=True)

    # Calculate the CDFs of u and v using their weights, if specified.
    u_cdf = u_cdf_indices.float() / u_values.size(0)
    v_cdf = v_cdf_indices.float() / v_values.size(0)

    # Compute the value of the integral based on the CDFs.
    # If p = 1 or p = 2, we avoid using torch.pow, which introduces an overhead
    # of about 15%.
    cdf_diff = torch.abs(u_cdf - v_cdf)
    if p == 1:
        return torch.sum(cdf_diff * deltas)
    if p == 2:
        return torch.sqrt(torch.sum((cdf_diff**2) * deltas)).item()
    return torch.pow(torch.sum((cdf_diff**p) * deltas), 1 / p).item()


def compute_average_wasserstein(
    num_particles: int,
    channels: int,
    gen_samples: Tensor,
    gt_samples: Tensor,
    p: int = 1,
    method: str = "custom",
) -> dict:
    """Computes the average wasserstein distance for every channel dimension"""

    avg_wass = {}  # Save the average wasserstein distance for every channel

    if method == "custom":
        for channel in range(channels):
            wass = []
            for sample in range(num_particles):
                gen_sample = gen_samples[:, sample, channel]
                gt_sample = gt_samples[:, sample, channel]
                wass.append(wasserstein_distance_1d(gen_sample, gt_sample))
            # compute the avg wasserstein for channel i
            avg_wass[f"wass_{channel}"] = torch.mean(torch.as_tensor(wass)).item()

    elif method == "scipy":
        for channel in range(channels):
            wass = []
            for sample in range(num_particles):
                gen_sample = gen_samples[:, sample, channel].cpu().numpy()
                gt_sample = gt_samples[:, sample, channel].cpu().numpy()
                wass.append(wasserstein_distance(gen_sample, gt_sample))
            avg_wass[f"wass_{channel}"] = float(np.mean(wass))
    else:
        raise ValueError(f"Wrong method, {method} does not exist.")

    return avg_wass
