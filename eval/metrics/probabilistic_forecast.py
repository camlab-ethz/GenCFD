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

"""Commonly-used metrics for probabilistic forecasting tasks."""

import torch

Tensor = torch.Tensor


def relative_L2_norm(
    gen_tensor: Tensor, gt_tensor: Tensor, axis: tuple | int
) -> Tensor:
    """Compute the relative L2 norm channel wise"""

    squared_mean_error = torch.mean((gt_tensor - gen_tensor) ** 2, dim=axis)
    squared_mean_gt = torch.mean(gt_tensor**2, dim=axis)
    return torch.sqrt(squared_mean_error / (squared_mean_gt + 1e-8))


def absolute_L2_norm(
    gen_tensor: Tensor, gt_tensor: Tensor, axis: tuple | int
) -> Tensor:
    """Compute the relative L2 norm channel wise"""

    squared_mean_error = torch.mean((gt_tensor - gen_tensor) ** 2, axis)
    return torch.sqrt(squared_mean_error)
