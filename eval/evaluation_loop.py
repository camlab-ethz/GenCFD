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
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval.metrics.stats_recorder import StatsRecorder
from dataloader.dataset import TrainingSetBase
from utils.dataloader_utils import normalize, denormalize
from utils.model_utils import reshape_jax_torch
from utils.eval_utils import summarize_metric_results
from utils.visualization_utils import plot_2d_sample, gen_gt_plotter_3d
from diffusion.samplers import Sampler


def run(
    *,
    sampler: Sampler,
    buffers: dict,
    monte_carlo_samples: int,
    stats_recorder: StatsRecorder,
    dataloader: DataLoader,
    dataset: TrainingSetBase,
    dataset_module: str,
    time_cond: bool,
    compute_metrics: bool = False,
    visualize: bool = False,
    device: torch.device = None,
    save_dir: str = None

) -> None:
    """Run benchmark evaluation on the specified dataset.

    This function performs benchmark evaluation in batches using the 
    provided denoising sampler. It can compute metrics through a Monte Carlo 
    simulation and optionally visualize results.

    Args:
        sampler (Sampler): The denoising-based diffusion sampler used for inference.
        buffers (dict): Buffers stored during training, including normalization arrays and tensors.
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
    if dataset_module not in [
        'ConditionalDataIC_Vel', 'ConditionalDataIC_Cloud_Shock_2D',
        'ConditionalDataIC_3D', 'ConditionalDataIC_3D_TG'
    ]:
        raise ValueError(f"To compute statistics use a conditional dataset, not {dataset_module}!")
    
    # To store either visualization or metric results a save_dir needs to be specified
    cwd = os.getcwd()
    save_dir = os.path.join(cwd, 'outputs' if save_dir is None else save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created a directory to store metrics and visualizations: {save_dir}")

    if compute_metrics:
        print("Compute Metrics")
        print(" ")
        # initialize the dataloader where the samples are drawn from a uniform discrete distribution
        dataloader = iter(dataloader)
        
        # Run a monte carlo simulation with a defined number of samples
        n_iter = monte_carlo_samples // batch_size
        # initialize stats_recorder to keep track of metrics
        
        # for i in range(n_iter):
        for i in tqdm(range(n_iter), desc="Evaluating Monte Carlo Samples"):
            # run n_iter number of iterations
            batch = next(dataloader)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            u0 = batch['initial_cond']
            u = batch['target_cond']
            if time_cond:
                lead_time = batch['lead_time'].reshape(-1,1)
            else:
                lead_time = [None] * batch_size

            # normalize inputs (initial conditions) and outputs (solutions)
            # u0_norm = reshape_jax_torch(dataset.normalize_input(reshape_jax_torch(u0)))
            u0_norm = reshape_jax_torch(
                normalize(
                    reshape_jax_torch(u0),
                    mean=buffers['mean_training_input'],
                    std=buffers['std_training_input']
                )
            )
            
            gen_batch = torch.empty(u.shape, device=device)
            for batch in range(batch_size):
                gen_sample = sampler.generate(
                    num_samples=1, 
                    y=u0_norm[batch, ...].unsqueeze(0), 
                    lead_time=lead_time[batch]
                ).detach()

                gen_batch[batch] = gen_sample.squeeze(0)
            # update relevant metrics and denormalize the generated results
            # u_gen = reshape_jax_torch(dataset.denormalize_output(reshape_jax_torch(gen_batch)))
            u_gen = reshape_jax_torch(
                denormalize(
                    reshape_jax_torch(gen_batch),
                    mean=buffers['mean_training_output'],
                    std=buffers['std_training_output']
                )
            )

            # solutions are stored with shape (bs, c, z, y, x)
            stats_recorder.update_step(u_gen, u)
        
        summarize_metric_results(stats_recorder, save_dir)

    if visualize:
        # Run a single run to visualize results without computing metrics
        batch = next(iter(dataloader)) # uniform random distribution
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        u0 = batch['initial_cond']
        u = batch['target_cond']

        if time_cond:
            lead_time = batch['lead_time'].reshape(-1,1)
        else:
            lead_time = [None] * batch_size

        # normalize inputs (initial conditions) and outputs (solutions)
        u0_norm = reshape_jax_torch(
            normalize(
                reshape_jax_torch(u0),
                mean=buffers['mean_training_input'],
                std=buffers['std_training_input']
            )
        )
        
        gen_batch = torch.empty(u.shape, device=device)
        for batch in range(batch_size):
            gen_sample = sampler.generate(
                num_samples=1, 
                y=u0_norm[batch, ...].unsqueeze(0), 
                lead_time=lead_time[batch]
            ).detach()

            gen_batch[batch] = gen_sample.squeeze(0)
        # update relevant metrics and denormalize the generated results
        # u_gen = reshape_jax_torch(dataset.denormalize_output(reshape_jax_torch(gen_batch)))
        u_gen = reshape_jax_torch(
            denormalize(
                reshape_jax_torch(gen_batch),
                mean=buffers['mean_training_output'],
                std=buffers['std_training_output']
            )
        )

        ndim = u_gen.ndim
        if ndim == 4:
            # plot 2D results
            plot_2d_sample(
                gen_sample=u_gen[0], gt_sample=u[0], 
                axis=0, save=True, save_dir=save_dir
            )
        elif ndim == 5:
            # plot 3D results
            gen_gt_plotter_3d(
                gt_sample=u[0], gen_sample=u_gen[0], axis=0, 
                save=True, save_dir=save_dir
            )