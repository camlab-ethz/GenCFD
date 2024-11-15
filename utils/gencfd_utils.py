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

from argparse import ArgumentParser
from typing import Tuple, Sequence, Dict, Callable
import torch
import os
import re
import json
from torch import nn
from torch.utils.data import DataLoader, random_split

from model.building_blocks.unets.unets import UNet, PreconditionedDenoiser
from model.building_blocks.unets.unets3d import UNet3D, PreconditionedDenoiser3D
from model.probabilistic_diffusion.denoising_model import DenoisingModel
from utils.model_utils import get_model_args, get_denoiser_args
from utils.diffusion_utils import (
    get_diffusion_scheme,
    get_noise_sampling,
    get_noise_weighting,
    get_sampler_args,
    get_time_step_scheduler
)
from diffusion.diffusion import (
    NoiseLevelSampling, 
    NoiseLossWeighting
)
from dataloader.dataset import (
    TrainingSetBase,
    DataIC_Vel,
    DataIC_3D_Time,
    DataIC_3D_Time_TG,
    ConditionalDataIC_Vel,
    ConditionalDataIC_3D,
    ConditionalDataIC_3D_TG
)
from utils.callbacks import Callback ,TqdmProgressBar, TrainStateCheckpoint
from diffusion.samplers import SdeSampler, Sampler
from solvers.sde import EulerMaruyama

Tensor = torch.Tensor
TensorMapping = Dict[str, Tensor]
DenoiseFn = Callable[[Tensor, Tensor, TensorMapping | None], Tensor]

# ***************************
# Load Dataset and Dataloader
# ***************************

def get_dataset(
        name: str, 
        device: torch.device = None,
        is_time_dependent: bool = False
    ) -> TrainingSetBase:
    """Returns the correct dataset and if the dataset has a time dependency
    This is necessary for the evaluation pipeline if there is no json file 
    provided.
    """

    if name == 'DataIC_Vel':
        dataset = DataIC_Vel(device=device)
        time_cond = False
    
    elif name == 'DataIC_3D_Time':
        dataset = DataIC_3D_Time(device=device)
        time_cond = True

    elif name == 'DataIC_3D_Time_TG':
        dataset = DataIC_3D_Time_TG(device=device)
        time_cond = True
    
    elif name == 'ConditionalDataIC_Vel':
        dataset = ConditionalDataIC_Vel(device=device)
        time_cond = False
    
    elif name == 'ConditionalDataIC_3D':
        dataset = ConditionalDataIC_3D(device=device)
        time_cond = True
    
    elif name == 'ConditionalDataIC_3D_TG':
        dataset = ConditionalDataIC_3D_TG(device=device)
        time_cond = True
    
    else:
        raise ValueError(f"Dataset {name} doesn't exist")
    
    if is_time_dependent:
        return dataset, time_cond
    else:
        return dataset


def get_dataset_loader(
        name: str, 
        batch_size: int = 5,
        num_worker: int = 0,
        split: bool = True, 
        split_ratio: float = 0.8,
        rng: torch.Generator = None,
        device: torch.device = None
    ) -> Tuple[DataLoader, DataLoader] | DataLoader:
    """Return a training and evaluation dataloader or a single dataloader"""

    dataset = get_dataset(name=name, device=device)

    if split:
        train_size = int(split_ratio * len(dataset))
        eval_size = len(dataset) - train_size
        train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size], rng)
        train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_worker,
            generator=rng
        )
        eval_dataloader = DataLoader(
            dataset=eval_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_worker,
            generator=rng
        )
        return (train_dataloader, eval_dataloader)
    else:
        return DataLoader(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_worker,
            generator=rng
        )
    
# ***************************
# Load Denoiser
# ***************************

def get_model(
        args: ArgumentParser, 
        out_channels: int, 
        rng: torch.Generator,
        device: torch.device = None,
        buffer_dict: dict = None,
        dtype: torch.dtype = torch.float32
    ) -> nn.Module:
    """Get the correct model"""
    
    model_args = get_model_args(
        args=args, out_channels=out_channels, 
        rng=rng, device=device,
        buffer_dict=buffer_dict,
        dtype=dtype
    )

    if args.model_type == 'UNet':
        return UNet(**model_args)
    
    elif args.model_type == 'PreconditionedDenoiser':
        return PreconditionedDenoiser(**model_args)
    
    elif args.model_type == 'UNet3D':
        return UNet3D(**model_args)
    
    elif args.model_type == 'PreconditionedDenoiser3D':
        return PreconditionedDenoiser3D(**model_args)
    
    
    else:
        raise ValueError(f"Model {args.model_type} does not exist")
    

def get_denoising_model(
        args: ArgumentParser,
        input_shape: int,
        input_channels: int,
        denoiser: nn.Module,
        noise_sampling: NoiseLevelSampling,
        noise_weighting: NoiseLossWeighting,
        rng: torch.Generator,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32
    ) -> DenoisingModel:
    """Create and retrieve the denoiser"""

    denoiser_args = get_denoiser_args(
        args=args,
        input_shape=input_shape,
        input_channels=input_channels,
        denoiser=denoiser,
        noise_sampling=noise_sampling,
        noise_weighting=noise_weighting,
        rng=rng,
        device=device,
        dtype=dtype
    )
    
    return DenoisingModel(**denoiser_args)



def create_denoiser(
        args: ArgumentParser, 
        input_shape: int,
        input_channels: int,
        out_channels: int,
        rng: torch.Generator,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        buffer_dict: dict = None
    ):
    """Get the denoiser and sampler if required"""

    model = get_model(
        args, out_channels, 
        rng, device,
        buffer_dict=buffer_dict,
        dtype=dtype
    )

    noise_sampling = get_noise_sampling(args, device)
    noise_weighting = get_noise_weighting(args, device)

    denoising_model = get_denoising_model(
        args=args,
        input_shape=input_shape,
        input_channels=input_channels,
        denoiser=model,
        noise_sampling=noise_sampling,
        noise_weighting=noise_weighting,
        rng=rng,
        device=device,
        dtype=dtype
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
        )
    ]

    if args.checkpoints:
        callbacks.append(
            TrainStateCheckpoint(
                base_dir= save_dir,
                save_every_n_step=args.save_every_n_steps
            )
        )
    
    return tuple(callbacks)


def get_latest_checkpoint(folder_path: str):
    """By specifying a folder path where all the checkpoints are stored 
    the latest model can be found!
    
    argument: folder_path passed as a string
    return: model path to the latest model
    """

    checkpoint_models = [
        f for f in os.listdir(folder_path)
    ]

    latest_checkpoint = max(
        checkpoint_models,
        key=lambda f: int(re.search(r'(\d+)', f).group())
    )

    return os.path.join(folder_path, latest_checkpoint)


def save_json_file(
        args: ArgumentParser, 
        time_cond: bool, 
        split_ratio: float,
        input_shape: tuple[int],
        out_shape: tuple[int],
        input_channel: int,
        output_channel: int,
        device: torch.device = None,
        seed: int = None
    ):
    """Create the training configuration file to use it later for inference"""

    config = {
        # general arguments
        "save_dir": args.save_dir,
        # dataset arguments
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "split_ratio": split_ratio,
        "worker": args.worker,
        "time_cond": time_cond,
        "input_shape": input_shape,
        "out_shape": out_shape,
        "input_channel": input_channel,
        "output_channel": output_channel,
        # model arguments
        "model_type": args.model_type,
        "num_heads": args.num_heads,
        # training arguments
        "use_mixed_precision": args.use_mixed_precision,
        "num_train_steps": args.num_train_steps,
        "task": args.task,
        "device": str(device) if device is not None else None,
        "seed": seed,
    }

    config_path = os.path.join(args.save_dir, "training_config.json")
    os.makedirs(args.save_dir, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"Training configuration saved to {config_path}")


def load_json_file(config_path: str):
    """Load the training configurations from a JSON file."""

    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(
            f"Configuration file not found at {config_path}. Using passed arguments"
        )
        return None


def replace_args(args: ArgumentParser, train_args: dict):
    """Replace parser arguments with used arguments during training.
    There is a skip list to avoid that every argument gets replaced."""

    skip_list = ["dataset", "save_dir", "batch_size"]

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
        rng: torch.Generator,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32
    ) -> Sampler:

    scheme = get_diffusion_scheme(args, device)

    integrator = EulerMaruyama(
        rng=rng, 
        time_axis_pos=args.time_axis_pos,
        terminal_only=args.terminal_only
    )

    tspan = get_time_step_scheduler(args=args, scheme=scheme, device=device, dtype=dtype)

    sampler_args = get_sampler_args(
        args=args,
        input_shape=input_shape,
        scheme=scheme,
        denoise_fn=denoise_fn,
        tspan=tspan,
        integrator=integrator,
        rng=rng,
        device=device,
        dtype=dtype
    )
    
    return SdeSampler(**sampler_args)