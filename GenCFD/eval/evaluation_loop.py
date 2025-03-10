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

"""Run Inference loops to generate statistical metrics or visualize results."""
import os
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from GenCFD.eval.metrics.stats_recorder import StatsRecorder
from GenCFD.dataloader.dataset import TrainingSetBase
from GenCFD.utils.dataloader_builder import normalize, denormalize
from GenCFD.utils.model_utils import reshape_jax_torch
from GenCFD.utils.eval_utils import summarize_metric_results
from GenCFD.utils.visualization_utils import plot_2d_sample, gen_gt_plotter_3d
from GenCFD.diffusion.samplers import Sampler


def run(
    *,
    sampler: Sampler,
    monte_carlo_samples: int,
    stats_recorder: StatsRecorder,
    # Dataset configs
    dataloader: DataLoader,
    dataset: TrainingSetBase,
    dataset_module: str,
    time_cond: bool,
    # Eval configs
    compute_metrics: bool = False,
    visualize: bool = False,
    save_gen_samples: bool = False,
    device: torch.device = None,
    save_dir: str = None,
    # DDP configs
    world_size: int = 1,
    local_rank: int = -1,
) -> None:
    """Run benchmark evaluation on the specified dataset.

    This function performs benchmark evaluation in batches using the
    provided denoising sampler. It can compute metrics through a Monte Carlo
    simulation and optionally visualize results.

    Args:
        sampler (Sampler): The denoising-based diffusion sampler used for inference.
        monte_carlo_samples (int): The number of Monte Carlo samples to use for metric computation,
            helping to mitigate computational demand during inference.
        stats_recorder (StatsRecorder): An object for recording evaluation statistics.
        dataloader (DataLoader): Initialized PyTorch DataLoader for batching the dataset.
        dataset (TrainingSetBase): The dataset class containing input and output channels.
        dataset_module (str): The name of the dataset module being used.
        time_cond (bool): Flag indicating whether the dataset has a time dependency.
        compute_metrics (bool, optional): If True, performs the Monte Carlo simulation to compute and
            store metrics in the specified directory. Defaults to False.
        visualize (bool, optional): If True, renders samples for 3D datasets or plots for 2D datasets.
            Defaults to False.
        device (torch.device, optional): The device on which to run the evaluation, either 'cuda' or 'cpu'.
        save_dir (str, optional): Path to the directory where metrics and visualization results will be saved.

    Returns:
        None
    """
    batch_size = dataloader.batch_size

    # first check if the correct dataset is used to compute statistics
    # if dataset_module not in [
    #     'ConditionalDataIC_Vel', 'ConditionalDataIC_Cloud_Shock_2D',
    #     'ConditionalDataIC_3D', 'ConditionalDataIC_3D_TG'
    # ]:
    #     if (world_size > 1 and local_rank == 0) or world_size == 1:
    #         raise ValueError(f"To compute statistics use a conditional dataset, not {dataset_module}!")

    # To store either visualization or metric results a save_dir needs to be specified
    cwd = os.getcwd()
    save_dir = os.path.join(cwd, "outputs" if save_dir is None else save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        if (world_size > 1 and local_rank == 0) or world_size == 1:
            print(
                f"Created a directory to store metrics and visualizations: {save_dir}"
            )

    if compute_metrics:
        if (world_size > 1 and local_rank == 0) or world_size == 1:
            print("Compute Metrics")
            print(" ")

        # initialize the dataloader where the samples are drawn from a uniform discrete distribution
        dataloader = iter(dataloader)

        # Run a monte carlo simulation with a defined number of samples
        n_iter = (monte_carlo_samples // batch_size) // world_size

        if (world_size > 1 and local_rank == 0) or world_size == 1:
            progress_bar = tqdm(
                total=monte_carlo_samples, desc="Evaluating Monte Carlo Samples"
            )

        # for i in range(n_iter):
        # for i in tqdm(range(n_iter), desc="Evaluating Monte Carlo Samples"):
        for i in range(n_iter):
            # run n_iter number of iterations
            batch = next(dataloader)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            u0 = batch["initial_cond"]
            u = batch["target_cond"]
            if time_cond:
                lead_time = batch["lead_time"]
            else:
                lead_time = [None] * batch_size

            gen_samples = sampler.generate(
                num_samples = batch_size,
                y=u0,
                lead_time=lead_time
            ).detach()

            # mask = dataset.get_mask().unsqueeze(0).unsqueeze(0).cuda()
            # u = u * mask
            stats_recorder.update_step(gen_samples, u)

            if (world_size > 1 and local_rank == 0) or world_size == 1:
                progress_bar.update(batch_size * world_size)

        if world_size > 1:
            # Aggregate and compute additional results
            dist.barrier(device_ids=[local_rank])
            stats_recorder.aggregate_all_processes()
            dist.barrier(device_ids=[local_rank])
            stats_recorder.gather_all_samples()
            dist.barrier(device_ids=[local_rank])

        if (world_size > 1 and local_rank == 0) or world_size == 1:
            summarize_metric_results(stats_recorder, save_dir)
            np.savez(
                os.path.join(save_dir, "physical_stats.npz"),
                mean_gen = stats_recorder.mean_gen.cpu().numpy(),
                mean_gt = stats_recorder.mean_gt.cpu().numpy(),
                std_gen = stats_recorder.std_gen.cpu().numpy(),
                std_gt = stats_recorder.std_gt.cpu().numpy(),
                gen_samples = stats_recorder.gen_samples.cpu().numpy(),
                gt_samples = stats_recorder.gt_samples.cpu().numpy()
            )
            if stats_recorder.compute_cfd_metrics:
                np.savez(
                    os.path.join(save_dir,"spectral_stats.npz"), 
                    epsilon_gen=stats_recorder.epsilon_gen.cpu().numpy(),
                    epsilon_gt=stats_recorder.epsilon_gt.cpu().numpy(),
                    spectrum_gen = stats_recorder.spectrum_gen.cpu().numpy(),
                    spectrum_gt = stats_recorder.spectrum_gt.cpu().numpy()
                )

    if visualize or save_gen_samples:
        # Run a single run to visualize results without computing metrics
        batch = next(iter(dataloader))  # uniform random distribution
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        u0 = batch["initial_cond"]
        u = batch["target_cond"]

        if time_cond:
            lead_time = batch['lead_time']
        else:
            lead_time = [None] * batch_size

        gen_samples = sampler.generate(
            num_samples = batch_size,
            y=u0,
            lead_time=lead_time
        ).detach()

        # IF Masking is required
        # mask = dataset.get_mask().unsqueeze(0).unsqueeze(0).cuda()
        # u = u * mask

        if save_gen_samples and (local_rank == 0 or local_rank == -1):
            # Put results on CPU first before storing them
            u_gen_np = gen_samples.cpu().numpy()
            u_np = u.cpu().numpy()

            # Save the arrays to an .npz file
            save_path = os.path.join(save_dir, "generated_samples.npz")
            np.savez(save_path, gen_sample=u_gen_np, gt_sample=u_np)
            print(
                f"Samples drawn from a uniform distribution are stored as {save_path}"
            )

        elif visualize and (local_rank == 0 or local_rank == -1):
            # Visualize the results instead!
            ndim = gen_samples.ndim
            if ndim == 4:
                # plot 2D results
                plot_2d_sample(
                    gen_sample=gen_samples[-1],
                    gt_sample=u[-1],
                    axis=0,
                    save=True,
                    save_dir=save_dir,
                )
            elif ndim == 5:
                # plot 3D results
                gen_gt_plotter_3d(
                    gt_sample=u[0],
                    gen_sample=gen_samples[0],
                    axis=0,
                    save=True,
                    save_dir=save_dir,
                )
