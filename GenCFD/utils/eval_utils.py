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

import os
import json

from GenCFD.eval.metrics.stats_recorder import StatsRecorder
from GenCFD.eval.metrics.probabilistic_forecast import relative_L2_norm, absolute_L2_norm
from GenCFD.eval.metrics.wasserstein import compute_average_wasserstein


def summarize_metric_results(
    stats_recorder: StatsRecorder,
    save_dir: str,
    output_file: str = "metrics_results.json",
) -> dict:
    """
    Summarizes the evaluation metrics and stores them in a JSON file.

    Parameters:
    -----------
    stats_recorder : StatsRecorder
        Object that contains accumulated metrics for ground truth and generated data.
    save_dir : str
        Directory to store the JSON file.
    output_file : str
        Filename of the JSON file.

    Returns:
    --------
    metrics_dict : dict
        Dictionary containing the summarized metrics.
    """

    # Compute relative and absolute L2 norms
    rel_mean = relative_L2_norm(
        gen_tensor=stats_recorder.mean_gen,
        gt_tensor=stats_recorder.mean_gt,
        axis=stats_recorder.axis,
    ).tolist()

    rel_std = relative_L2_norm(
        gen_tensor=stats_recorder.std_gen,
        gt_tensor=stats_recorder.std_gt,
        axis=stats_recorder.axis,
    ).tolist()

    abs_mean = absolute_L2_norm(
        gen_tensor=stats_recorder.mean_gen,
        gt_tensor=stats_recorder.mean_gt,
        axis=stats_recorder.axis,
    ).tolist()

    abs_std = absolute_L2_norm(
        gen_tensor=stats_recorder.std_gen,
        gt_tensor=stats_recorder.std_gt,
        axis=stats_recorder.axis,
    ).tolist()

    # Monte Carlo sampled metrics
    gen_monte_carlo_samples = stats_recorder.gen_samples
    gt_monte_carlo_samples = stats_recorder.gt_samples

    # Compute average Wasserstein distances
    wasserstein_distance_torch = compute_average_wasserstein(
        monte_carlo_samples=stats_recorder.monte_carlo_samples,
        channels=stats_recorder.channels,
        gen_samples=gen_monte_carlo_samples,
        gt_samples=gt_monte_carlo_samples,
        p=1,
        method="custom",
    )

    # Construct the metrics dictionary
    metrics_dict = {
        "mean": {
            "relative": rel_mean,
            "absolute": abs_mean,
        },
        "std": {
            "relative": rel_std,
            "absolute": abs_std,
        },
        "wasserstein_distance": wasserstein_distance_torch,
    }

    # Print metrics to ensure they're logged regardless of save status
    print("Metric results:")
    print(json.dumps(metrics_dict, indent=4))

    # Save to JSON file
    save_file = os.path.join(save_dir, output_file)
    try:
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
        with open(save_file, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"Metrics successfully saved to {save_file}")

    except Exception as e:
        print(f"Failed to save metrics to {save_file}: {e}")

    return metrics_dict
