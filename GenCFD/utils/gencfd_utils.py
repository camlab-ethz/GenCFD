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

"""Utilities for train_gencfd and evaluate_gencfd"""

from argparse import ArgumentParser
from typing import Tuple, Sequence, Dict, Callable
import torch
import os
import re
import json
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from GenCFD.model.building_blocks.unets.unets import UNet, PreconditionedDenoiser
from GenCFD.model.building_blocks.unets.unets3d import UNet3D, PreconditionedDenoiser3D
from GenCFD.model.probabilistic_diffusion.denoising_model import DenoisingModel
from GenCFD.utils.model_utils import get_model_args
from GenCFD.utils.denoiser_utils import get_denoiser_args
from GenCFD.utils.diffusion_utils import (
    get_diffusion_scheme,
    get_noise_sampling,
    get_noise_weighting,
    get_sampler_args,
    get_time_step_scheduler,
)
from GenCFD.diffusion.diffusion import NoiseLevelSampling, NoiseLossWeighting
from GenCFD.dataloader.dataset import (
    train_test_split,
    TrainingSetBase,
)
from GenCFD.dataloader.fluid_flows_3d import (
    ShearLayer3D,
    TaylorGreen3D,
    Nozzle3D,
    ConditionalShearLayer3D,
    ConditionalTaylorGreen3D,
    ConditionalNozzle3D,
)
from GenCFD.dataloader.metadata import METADATA_CLASSES
from GenCFD.utils.callbacks import Callback, TqdmProgressBar, TrainStateCheckpoint
from GenCFD.diffusion.samplers import SdeSampler, Sampler
from GenCFD.solvers.sde import EulerMaruyama

Tensor = torch.Tensor
TensorMapping = Dict[str, Tensor]
DenoiseFn = Callable[[Tensor, Tensor, TensorMapping | None], Tensor]

# ***************************
# Load Dataset and Dataloader
# ***************************


def get_dataset(
    name: str,
    is_time_dependent: bool = False,
    # device: torch.device = None
) -> TrainingSetBase:
    """
    Returns the appropriate dataset and its time dependency status.

    This function selects and initializes the correct dataset based on the provided name.
    It also returns whether the dataset is time-dependent, which is crucial for the
    evaluation pipeline when no JSON configuration file is provided.

    Note:
        The `lead_time=True` setting is applied explicitly to all datasets here, even
        though for 2D datasets this would typically be `False`. This ensures consistent
        behavior across datasets, particularly for 3D cases where time dependency is
        always required.

    Args:
        name (str): The name of the dataset to load. Must be one of the predefined datasets.
        is_time_dependent (bool, optional): If `True`, the function will return the
            dataset along with its time dependency status. Defaults to `False`.

    Returns:
        TrainingSetBase: The initialized dataset.
        tuple[TrainingSetBase, bool]: If `is_time_dependent=True`, returns a tuple of
            the dataset and a boolean indicating if it is time-dependent.

    Raises:
        ValueError: If the provided dataset name is not recognized.
    """

    metadata = METADATA_CLASSES[name]

    if name == "ShearLayer3D":
        dataset = ShearLayer3D(metadata=metadata)
        time_cond = True

    elif name == "TaylorGreen3D":
        dataset = TaylorGreen3D(metadata=metadata)
        time_cond = True

    elif name == "Nozzle3D":
        dataset = Nozzle3D(metadata=metadata)
        time_cond = True

    elif name == "ConditionalShearLayer3D":
        dataset = ConditionalShearLayer3D(metadata=metadata)
        time_cond = True

    elif name == "ConditionalTaylorGreen3D":
        dataset = ConditionalTaylorGreen3D(metadata=metadata)
        time_cond = True

    elif name == "ConditionalNozzle3D":
        dataset = ConditionalNozzle3D(metadata=metadata)
        time_cond = True

    else:
        raise ValueError(f"Dataset {name} doesn't exist")

    if is_time_dependent:
        return dataset, time_cond
    else:
        return dataset


def get_distributed_sampler(
    args: ArgumentParser, dataset: TrainingSetBase
) -> DistributedSampler:
    """
    For DDP a Distributed Sampler is requited where
    each process gets a unique subset of data in a way that
    there is no overlap between subsets
    """
    dist_sampler = DistributedSampler(
        dataset, rank=args.local_rank, num_replicas=dist.get_world_size(), shuffle=True
    )
    return dist_sampler


def get_dataset_loader(
    args: ArgumentParser,
    name: str,
    batch_size: int = 5,
    num_worker: int = 0,
    prefetch_factor: int = 2,  # default DataLoader value
    split: bool = True,
    split_ratio: float = 0.9,
) -> Tuple[DataLoader, DataLoader] | DataLoader:
    """Return a training and evaluation dataloader or a single dataloader"""

    # is_time_dependent passes the bool time_cond and tells if the problem is time
    # dependent or not
    dataset, time_cond = get_dataset(name=name, is_time_dependent=True)

    if args.world_size > 1:
        dist_sampler = get_distributed_sampler(args, dataset)

    if split:
        # split the dataset into train and eval
        train_dataset, eval_dataset = train_test_split(dataset, split_ratio=split_ratio)

        if args.world_size > 1:
            train_sampler = get_distributed_sampler(args, train_dataset)
            eval_sampler = get_distributed_sampler(args, eval_dataset)

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True if args.world_size == 1 else False,
            pin_memory=True,
            num_workers=num_worker,
            prefetch_factor=prefetch_factor,
            sampler=train_sampler if args.world_size > 1 else None,
        )
        eval_dataloader = DataLoader(
            dataset=eval_dataset,
            batch_size=batch_size,
            shuffle=True if args.world_size == 1 else False,
            pin_memory=True,
            num_workers=num_worker,
            prefetch_factor=prefetch_factor,
            sampler=eval_sampler if args.world_size > 1 else None,
        )
        return (train_dataloader, eval_dataloader, dataset, time_cond)

    else:
        if args.world_size > 1:
            sampler = get_distributed_sampler(args, dataset)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True if args.world_size == 1 else False,
            pin_memory=True,
            num_workers=num_worker,
            prefetch_factor=prefetch_factor,
            sampler=sampler if args.world_size > 1 else None,
        )
        return (dataloader, dataset, time_cond)


def get_buffer_dict(
    dataset: TrainingSetBase, device: torch.device = None, create_dummy: bool = False
) -> dict:
    """Build the Buffer Dictionary with normalizatino parameters"""

    if create_dummy:
        mean_training_input = torch.zeros((dataset.input_channel,))
        mean_training_output = torch.zeros((dataset.output_channel,))
        std_training_input = torch.ones((dataset.input_channel,))
        std_training_output = torch.ones((dataset.output_channel,))
    else:
        mean_training_input = torch.tensor(
            dataset.mean_training_input, dtype=torch.float32, device=device
        )
        mean_training_output = torch.tensor(
            dataset.mean_training_output, dtype=torch.float32, device=device
        )
        std_training_input = torch.tensor(
            dataset.std_training_input, dtype=torch.float32, device=device
        )
        std_training_output = torch.tensor(
            dataset.std_training_output, dtype=torch.float32, device=device
        )

    buffer_dict = {
        "mean_training_input": mean_training_input,
        "mean_training_output": mean_training_output,
        "std_training_input": std_training_input,
        "std_training_output": std_training_output,
    }

    return buffer_dict


def adjust_keys(
    dictionary: dict, is_compiled: bool, is_parallelized: bool, use_ddp_wrapper: bool
) -> dict:
    """Function returns a dictionary and deletes the key prefixes
    which comes from compiling or evaluating the model in parallel"""

    keyword_compiled = "_orig_mod."
    keyword_ddp = "module."

    if is_compiled:
        # Delete the compilation keyword
        dictionary = {
            key.replace(keyword_compiled, ""): value
            for key, value in dictionary.items()
        }

    if is_parallelized and use_ddp_wrapper:
        # Delete the keyword when the model is parallelized
        dictionary = {
            key.replace(keyword_ddp, ""): value for key, value in dictionary.items()
        }

    return dictionary


# ***************************
# Load Denoiser
# ***************************


def get_model(
    args: ArgumentParser,
    in_channels: int,
    out_channels: int,
    spatial_resolution: tuple,
    time_cond: bool,
    device: torch.device = None,
    buffer_dict: dict = None,
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Get the correct model"""

    model_args = get_model_args(
        args=args,
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_resolution=spatial_resolution,
        time_cond=time_cond,
        device=device,
        buffer_dict=buffer_dict,
        dtype=dtype,
    )

    if args.model_type == "UNet":
        return UNet(**model_args)

    elif args.model_type == "PreconditionedDenoiser":
        return PreconditionedDenoiser(**model_args)

    elif args.model_type == "UNet3D":
        return UNet3D(**model_args)

    elif args.model_type == "PreconditionedDenoiser3D":
        return PreconditionedDenoiser3D(**model_args)

    else:
        raise ValueError(f"Model {args.model_type} does not exist")


def get_denoising_model(
    args: ArgumentParser,
    input_channels: int,
    spatial_resolution: Sequence[int],
    time_cond: bool,
    denoiser: nn.Module,
    noise_sampling: NoiseLevelSampling,
    noise_weighting: NoiseLossWeighting,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> DenoisingModel:
    """Create and retrieve the denoiser"""

    denoiser_args = get_denoiser_args(
        args=args,
        spatial_resolution=spatial_resolution,
        time_cond=time_cond,
        denoiser=denoiser,
        noise_sampling=noise_sampling,
        noise_weighting=noise_weighting,
        device=device,
        dtype=dtype,
    )

    return DenoisingModel(**denoiser_args)


def create_denoiser(
    args: ArgumentParser,
    input_channels: int,
    out_channels: int,
    spatial_resolution: Sequence[int],
    time_cond: bool,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    buffer_dict: dict = None,
    use_ddp_wrapper: bool = False,
):
    """Get the denoiser and sampler if required"""

    model = get_model(
        args=args,
        # For the UNet model input channels and output channels are concatenated
        in_channels=input_channels + out_channels,
        out_channels=out_channels,
        spatial_resolution=spatial_resolution,
        time_cond=time_cond,
        device=device,
        buffer_dict=buffer_dict,
        dtype=dtype,
    )

    if args.compile:
        model = torch.compile(model)

    if args.world_size > 1 and use_ddp_wrapper:
        model = DDP(model, device_ids=[args.local_rank])

    if args.local_rank == 0 or args.local_rank == -1:
        print(" ")
        print(f"Compilation mode: {args.compile}, World Size: {args.world_size}")

    noise_sampling = get_noise_sampling(args, device)
    noise_weighting = get_noise_weighting(args, device)

    denoising_model = get_denoising_model(
        args=args,
        input_channels=input_channels,
        spatial_resolution=spatial_resolution,
        time_cond=time_cond,
        denoiser=model,
        noise_sampling=noise_sampling,
        noise_weighting=noise_weighting,
        device=device,
        dtype=dtype,
    )

    return denoising_model


# ***************************
# Get Callback Method
# ***************************


def create_callbacks(args: ArgumentParser, save_dir: str) -> Sequence[Callback]:
    """Get the callback methods like profilers, metric collectors, etc."""

    train_monitors = ["loss", "loss_std"]
    if args.track_memory:
        train_monitors.append("mem")

    callbacks = [
        TqdmProgressBar(
            total_train_steps=args.num_train_steps,
            train_monitors=train_monitors,
            world_size=args.world_size,
            local_rank=args.local_rank,
        )
    ]

    if args.checkpoints:
        checkpoint_callback = TrainStateCheckpoint(
            base_dir=save_dir,
            save_every_n_step=args.save_every_n_steps,
            world_size=args.world_size,
            local_rank=args.local_rank,
        )
        callbacks.insert(0, checkpoint_callback)

    return tuple(callbacks)


def get_latest_checkpoint(folder_path: str):
    """By specifying a folder path where all the checkpoints are stored
    the latest model can be found!

    argument: folder_path passed as a string
    return: model path to the latest model
    """

    checkpoint_models = [f for f in os.listdir(folder_path)]

    if not checkpoint_models:
        return None

    latest_checkpoint = max(
        checkpoint_models, key=lambda f: int(re.search(r"(\d+)", f).group())
    )

    return os.path.join(folder_path, latest_checkpoint)


def save_json_file(
    args: ArgumentParser,
    time_cond: bool,
    split_ratio: float,
    out_shape: Sequence[int],
    input_channel: int,
    output_channel: int,
    spatial_resolution: Sequence[int],
    device: torch.device = None,
    seed: int = None,
):
    """Create the training configuration file to use it later for inference"""

    config = {
        # general arguments
        "save_dir": args.save_dir,
        "world_size": args.world_size,
        # dataset arguments
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "split_ratio": split_ratio,
        "worker": args.worker,
        "time_cond": time_cond,
        "out_shape": out_shape,
        "input_channel": input_channel,
        "output_channel": output_channel,
        "spatial_resolution": spatial_resolution,
        # model arguments
        "model_type": args.model_type,
        "compile": args.compile,
        "num_heads": args.num_heads,
        # training arguments
        "use_mixed_precision": args.use_mixed_precision,
        "num_train_steps": args.num_train_steps,
        "task": args.task,
        "device": device.type if device is not None else None,
        "seed": seed,
    }

    config_path = os.path.join(args.save_dir, "training_config.json")
    os.makedirs(args.save_dir, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(" ")
    print(f"Training configuration saved to {config_path}")


def load_json_file(config_path: str):
    """Load the training configurations from a JSON file."""

    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Configuration file not found at {config_path}. Using passed arguments")
        return None


def replace_args(args: ArgumentParser, train_args: dict):
    """Replace parser arguments with used arguments during training.
    There is a skip list to avoid that every argument gets replaced."""

    skip_list = [
        "dataset",
        "save_dir",
        "batch_size",
        "compile",
        "world_size",
    ]

    for key, value in train_args.items():
        if key in skip_list:
            continue
        if hasattr(args, key):
            setattr(args, key, value)


# ***************************
# Load Sampler
# ***************************


def create_sampler(
    args: ArgumentParser,
    input_shape: int,
    denoise_fn: DenoiseFn,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> Sampler:

    scheme = get_diffusion_scheme(args, device)

    integrator = EulerMaruyama(
        time_axis_pos=args.time_axis_pos, terminal_only=args.terminal_only
    )

    tspan = get_time_step_scheduler(
        args=args, scheme=scheme, device=device, dtype=dtype
    )

    sampler_args = get_sampler_args(
        args=args,
        input_shape=input_shape,
        scheme=scheme,
        denoise_fn=denoise_fn,
        tspan=tspan,
        integrator=integrator,
        device=device,
        dtype=dtype,
    )

    return SdeSampler(**sampler_args)
