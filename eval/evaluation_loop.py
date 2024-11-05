import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from eval.metrics.stats_recorder import StatsRecorder
from eval.metrics.probabilistic_forecast import relative_L2_norm, absolute_L2_norm
from dataloader.dataset import TrainingSetBase
from utils.model_utils import reshape_jax_torch
from utils.visualization_utils import plot_2d_sample, gen_gt_plotter_3d
from diffusion.samplers import Sampler


def run(
    *,
    sampler: Sampler,
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
    """Runs a benchmark evaluation.

    This function runs the benchmark evaluation in bunches, or "groups of
    batches" (where group size = `num_aggregation_batches`). After a group is done
    evaluating, the results are (optionally) saved/checkpointed. At the end of the
    whole evaluation (either by reaching `max_eval_batches` or the dataloader
    raising `StopIteration`), the aggregated metrics are saved.

    Args: 
    """
    batch_size = dataloader.batch_size
    # first check if the correct dataset is used to compute statistics
    if dataset_module not in ['ConditionalDataIC_Vel', 'ConditionalDataIC_3D']:
        raise ValueError(f"To compute statistics use a conditional dataset, not {dataset_module}!")
    
    # To store either visualization or metric results a save_dir needs to be specified
    cwd = os.getcwd()
    save_dir = os.path.join(cwd, 'outputs' if save_dir is None else save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created a directory to store metrics and visualizations: {save_dir}")
    
    # check whether problem requires a conditioning on the lead time
    if time_cond:
        # constant lead_time with 1 and a single step from an initial condition to 1
        lead_time = torch.ones((batch_size, 1), dtype=torch.float32, device=device)
    else:
        lead_time = [None] * batch_size

    if compute_metrics:
        print(" ")
        print("Compute Metrics")
        # initialize the dataloader where the samples are drawn from a uniform discrete distribution
        dataloader = iter(dataloader)
        
        # Run a monte carlo simulation with a defined number of samples
        n_iter = monte_carlo_samples // batch_size
        # initialize stats_recorder to keep track of metrics
        
        for i in tqdm(range(n_iter), desc="Evaluating Monte Carlo Samples"):
            # run n_iter number of iterations
            batch = next(dataloader)
            u0 = batch[:, :dataset.output_channel, ...].to(device=device)
            u = batch[:, dataset.output_channel:, ...].to(device=device)

            # normalize inputs (initial conditions) and outputs (solutions)
            u0_norm = reshape_jax_torch(dataset.normalize_input(reshape_jax_torch(u0)))
            
            gen_batch = torch.empty(u.shape, device=device)
            for batch in range(batch_size):
                gen_sample = sampler.generate(
                    num_samples=1, 
                    y=u0_norm[batch, ...].unsqueeze(0), 
                    lead_time=lead_time[batch]
                ).detach()

                gen_batch[batch] = gen_sample.squeeze(0)
            # update relevant metrics and denormalize the generated results
            u_gen = reshape_jax_torch(dataset.denormalize_output(reshape_jax_torch(gen_batch)))

            # solutions are stored with shape (bs, c, z, y, x)
            stats_recorder.update_step(u_gen, u)
        
        # Show results accumulated over the number of monte carlo samples
        mean_gt = stats_recorder.mean_gt
        mean_gen = stats_recorder.mean_gen

        std_gt = stats_recorder.std_gt
        std_gen = stats_recorder.std_gen

        rel_mean = relative_L2_norm(gen_tensor=mean_gen, gt_tensor=mean_gt, axis=stats_recorder.axis)
        rel_std = relative_L2_norm(gen_tensor=std_gen, gt_tensor=std_gt, axis=stats_recorder.axis)

        abs_mean = absolute_L2_norm(gen_tensor=mean_gen, gt_tensor=mean_gt, axis=stats_recorder.axis)
        abs_std = absolute_L2_norm(gen_tensor=std_gen, gt_tensor=std_gt, axis=stats_recorder.axis)

        print(" ")
        print("Relative RMSE for each metric and channel")
        print(f"Mean Metric: {rel_mean}    STD Metric {rel_std}")
        print(" ")
        print("Absolute RMSE for each metric and channel")
        print(f"Mean Metric: {abs_mean}    STD Metric {abs_std}")

        # save results!
        np.savez(
            os.path.join(save_dir, f'eval_results_{monte_carlo_samples}_samples.npz'), 
            rel_mean=rel_mean.cpu().numpy(), 
            rel_std=rel_std.cpu().numpy(),
            abs_mean=abs_mean.cpu().numpy(),
            abs_std=abs_std.cpu().numpy()
        )

    if visualize:
        # Run a single run to visualize results without computing metrics

        batch = next(dataloader)
        u0 = batch[:, :dataset.output_channel, ...].to(device=device)
        u = batch[:, dataset.output_channel:, ...].to(device=device)

        # normalize inputs (initial conditions) and outputs (solutions)
        u0_norm = reshape_jax_torch(dataset.normalize_input(reshape_jax_torch(u0)))
        
        gen_batch = torch.empty(u.shape, device=device)
        for batch in range(batch_size):
            gen_sample = sampler.generate(
                num_samples=1, 
                y=u0_norm[batch, ...].unsqueeze(0), 
                lead_time=lead_time[batch]
            ).detach()

            gen_batch[batch] = gen_sample.squeeze(0)
        # update relevant metrics and denormalize the generated results
        u_gen = reshape_jax_torch(dataset.denormalize_output(reshape_jax_torch(gen_batch)))

        ndim = u_gen.ndim
        if ndim == 4:
            # plot 2D results
            plot_2d_sample(gen_sample=u_gen[0], gt_sample=u[0], axis=0)
        elif ndim == 5:
            # plot 3D results
            gen_gt_plotter_3d(gt_sample=u[0], gen_sample=u_gen[0], axis=0)
