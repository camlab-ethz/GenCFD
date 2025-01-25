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

"""Main File to Run Inference.

Options are to compute statistical metrics or visualize results.
"""

import time
import os
import sys
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch import optim

from GenCFD.train.train_states import DenoisingModelTrainState
from GenCFD.train.trainers import DenoisingTrainer
from GenCFD.utils.parser_utils import inference_args
from GenCFD.utils.gencfd_utils import (
    get_dataset_loader,
    create_denoiser,
    create_sampler,
    get_latest_checkpoint,
    load_json_file,
    replace_args,
    get_buffer_dict,
    adjust_keys,
)
from GenCFD.eval.metrics.stats_recorder import StatsRecorder
from GenCFD.eval import evaluation_loop


torch.set_float32_matmul_precision("high")  # Better performance on newer GPUs!
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0

# Setting global seed for reproducibility
torch.manual_seed(SEED)  # For CPU operations
torch.cuda.manual_seed(SEED)  # For GPU operations
torch.cuda.manual_seed_all(SEED)  # Ensure all GPUs (if multi-GPU) are set


def init_distributed_mode(args):
    """Initialize a Distributed Data Parallel Environment"""

    args.local_rank = int(os.getenv("LOCAL_RANK", -1))  # Get from environment variable

    if args.local_rank == -1:
        raise ValueError(
            "--local_rank was not set. Ensure torchrun is used to launch the script."
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend="nccl", rank=args.local_rank, world_size=args.world_size
        )
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        dist.init_process_group(
            backend="gloo", rank=args.local_rank, world_size=args.world_size
        )
        device = torch.device("cpu")
        print(" ")

    print(f"DDP initialized with rank {args.local_rank} and device {device}.")

    return args, device


if __name__ == "__main__":

    # get arguments for inference
    args = inference_args()

    # Initialize distributed mode (if multi-GPU)
    if args.world_size > 1:
        args, device = init_distributed_mode(args)
    else:
        print(" ")
        print(f"Used device: {device}")

    cwd = os.getcwd()
    if args.model_dir is None:
        raise ValueError("Path to a trained model is not specified!")
    model_dir = os.path.join(cwd, args.model_dir, "checkpoints")
    if not os.path.exists(model_dir):
        if (args.world_size > 1 and args.local_rank == 0) or args.world_size == 1:
            raise ValueError(f"Wrong Path, {args.model_dir} doesn't exist!")

    # read configurations which were used to train the model
    train_args = load_json_file(
        os.path.join(cwd, args.model_dir, "training_config.json")
    )

    dataloader, dataset, time_cond = get_dataset_loader(
        args=args,
        name=args.dataset,
        batch_size=args.batch_size,
        num_worker=args.worker,
        # Default prefetch factor is 2
        prefetch_factor=2 if args.worker > 1 else None,
        split=False,
    )

    out_shape = dataset.output_shape
    spatial_resolution = dataset.spatial_resolution

    if train_args:
        # replace every argument from train_args besides the dataset name!
        replace_args(args, train_args)

        if (args.world_size > 1 and args.local_rank == 0) or args.world_size == 1:
            # Check if the arguments used for training are the same as the evaluation dataset
            assert spatial_resolution == tuple(train_args["spatial_resolution"]), (
                f"spatial_resolution should be {tuple(train_args['spatial_resolution'])} "
                f"and not {spatial_resolution}"
            )
            assert out_shape == tuple(
                train_args["out_shape"]
            ), f"out_shape should be {tuple(train_args['out_shape'])} and not {out_shape}"

    # Dummy buffer values, for initialization! Necessary to load the model parameters
    buffer_dict = get_buffer_dict(dataset=dataset, create_dummy=True)

    # the compute_dtype needs to be the same as used for the trained model!
    denoising_model = create_denoiser(
        args=args,
        input_channels=dataset.input_channel,
        out_channels=dataset.output_channel,
        spatial_resolution=spatial_resolution,
        time_cond=time_cond,
        device=device,
        dtype=args.dtype,
        buffer_dict=buffer_dict,
        use_ddp_wrapper=False,
    )

    with torch.no_grad():
        denoising_model.initialize(
            batch_size=args.batch_size,
            time_cond=time_cond,
            input_channels=dataset.input_channel,
            output_channels=dataset.output_channel,
        )

    # Print number of Parameters:
    model_params = sum(
        p.numel() for p in denoising_model.denoiser.parameters() if p.requires_grad
    )
    if (args.world_size > 1 and args.local_rank == 0) or args.world_size == 1:
        print(" ")
        print(f"Total number of model parameters: {model_params}")
        print(" ")

    # Rebuild the trainer used for training
    trainer = DenoisingTrainer(
        model=denoising_model,
        optimizer=optim.AdamW(
            denoising_model.denoiser.parameters(),
            lr=args.peak_lr,
            weight_decay=args.weight_decay,
        ),
        device=device,
        ema_decay=args.ema_decay,
        store_ema=False,
        track_memory=False,
        use_mixed_precision=args.use_mixed_precision,
        is_compiled=args.compile,
        world_size=args.world_size,
        local_rank=args.local_rank,
    )

    latest_model_path = get_latest_checkpoint(model_dir)

    if (args.world_size > 1 and args.local_rank == 0) or args.world_size == 1:
        print("Load Model Parameters")
        print(f"Latest Path used: {latest_model_path}")
        print(" ")

    trained_state = DenoisingModelTrainState.restore_from_checkpoint(
        latest_model_path,
        model=denoising_model.denoiser,
        optimizer=trainer.optimizer,
        is_compiled=trainer.is_compiled,
        is_parallelized=False,
        use_ema=True,  # load ema parameters instead
        device=device,
    )

    # Retrieve the normalization buffer (mean and std tensors)
    buffers = dict(denoising_model.denoiser.named_buffers())
    for key, tensor in buffers.items():
        if tensor is not None:  # put tensor on same device!
            buffers[key] = tensor.to(device=device)

    # If model is compiled or evaluated in parallel adjust the keyword name
    buffers = adjust_keys(
        buffers,
        is_compiled=args.compile,
        is_parallelized=True if args.world_size > 1 else False,
        use_ddp_wrapper=False,
    )

    # Construct the inference function
    denoise_fn = trainer.inference_fn_from_state_dict(
        trained_state,
        denoiser=denoising_model.denoiser,
        task=args.task,
        lead_time=time_cond,
    )

    # Create Sampler
    sampler = create_sampler(
        args=args, input_shape=out_shape, denoise_fn=denoise_fn, device=device
    )

    # compute the effective number of monte carlo samples if world_size is greater than 1
    if args.world_size > 1:
        if args.monte_carlo_samples % args.world_size != 0:
            if args.local_rank == 0:
                print(
                    "Number of monte carlo samples should be divisible through the number of processes used!"
                )

        effective_samples = (
            args.monte_carlo_samples // (args.world_size * args.batch_size)
        ) * (args.world_size * args.batch_size)

        if effective_samples <= 0:
            error_msg = (
                f"Invalid configuration: Number of Monte Carlo samples ({args.monte_carlo_samples}), "
                f"batch size ({args.batch_size}), and world size ({args.world_size}) result in zero effective samples. "
                f"Ensure monte_carlo_samples >= world_size * batch_size."
            )
            if args.local_rank == 0:
                print(error_msg)
            dist.barrier()
            dist.destroy_process_group()
            sys.exit(0)

    # Initialize stats_recorder to keep track of metrics
    stats_recorder = StatsRecorder(
        batch_size=args.batch_size,
        ndim=len(out_shape) - 1,
        channels=dataset.output_channel,
        data_shape=out_shape,
        monte_carlo_samples=(
            args.monte_carlo_samples
            if args.world_size <= 1
            else effective_samples // args.world_size
        ),
        num_samples=1000,
        device=device,
        world_size=args.world_size,
    )

    if (args.world_size > 1 and args.local_rank == 0) or args.world_size == 1:
        if args.compute_metrics:
            tot_samples = (
                args.monte_carlo_samples if args.world_size <= 1 else effective_samples
            )
            print(
                f"Run Evaluation Loop with {tot_samples} Monte Carlo Samples and Batch Size {args.batch_size}"
            )
        if args.visualize:
            print(f"Run Visualization Loop")

    start_train = time.time()

    if args.world_size > 1:
        dist.barrier(device_ids=[args.local_rank])

    evaluation_loop.run(
        sampler=sampler,
        buffers=buffers,
        monte_carlo_samples=(
            args.monte_carlo_samples if args.world_size <= 1 else effective_samples
        ),
        stats_recorder=stats_recorder,
        # Dataset configs
        dataloader=dataloader,
        dataset=dataset,
        time_cond=time_cond,
        # Eval configs
        compute_metrics=args.compute_metrics,
        visualize=args.visualize,
        save_gen_samples=args.save_gen_samples,
        device=device,
        save_dir=args.save_dir,
        # DDP configs
        world_size=args.world_size,
        local_rank=args.local_rank,
    )

    end_train = time.time()
    elapsed_train = end_train - start_train
    if (args.world_size > 1 and args.local_rank == 0) or args.world_size == 1:
        print(" ")
        print(f"Done evaluation. Elapsed time {elapsed_train / 3600} h")

    if dist.is_initialized():
        dist.destroy_process_group()
