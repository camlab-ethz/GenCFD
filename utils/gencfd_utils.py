from argparse import ArgumentParser
from typing import Tuple, Sequence
import torch
import os
from torch import nn
from torch.utils.data import DataLoader, random_split

from model.building_blocks.unets.unets import UNet, PreconditionedDenoiser
from model.probabilistic_diffusion.denoising_model import DenoisingModel
from utils.model_utils import get_model_args, get_denoiser_args
from utils.diffusion_utils import (
    get_diffusion_scheme,
    get_noise_sampling,
    get_noise_weighting
)
from diffusion.diffusion import (
    NoiseLevelSampling, 
    NoiseLossWeighting
)
from dataloader.dataset import (
    TrainingSetBase,
    DataIC_Vel,
    DataIC_Vel_Test,
)
from utils.callbacks import Callback ,TqdmProgressBar, TrainStateCheckpoint


# ***************************
# Load Dataset and Dataloader
# ***************************

def get_dataset(name: str) -> TrainingSetBase:
    """Return the correct dataset"""

    if name == 'DataIC_Vel':
        return DataIC_Vel()
    
    elif name == 'DataIC_Vel_Test':
        return DataIC_Vel_Test()
    

def get_dataset_loader(
        name: str, 
        batch_size: int = 5,
        num_worker: int = 0,
        split: bool = True, 
        split_ratio: float = 0.8
    ) -> Tuple[DataLoader, DataLoader] | DataLoader:
    """Return a training and evaluation dataloader or a single dataloader"""

    dataset = get_dataset(name=name)

    if split:
        train_size = int(0.8 * len(dataset))
        eval_size = len(dataset) - train_size
        train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
        train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_worker
        )
        eval_dataloader = DataLoader(
            dataset=eval_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_worker
        )
        return (train_dataloader, eval_dataloader)
    else:
        return DataLoader(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_worker
        )
    
# ***************************
# Load Denoiser and Sampler
# ***************************

def get_model(
        args: ArgumentParser, 
        out_channels: int, 
        rng: torch.Generator,
        device: torch.device = None
    ) -> nn.Module:
    """Get the correct model"""
    
    model_args = get_model_args(
        args=args, out_channels=out_channels, rng=rng, device=device
    )

    if args.model_type == 'UNet':
        return UNet(**model_args)
    
    elif args.model_type == 'PreconditionedDenoiser':
        return PreconditionedDenoiser(**model_args)
    
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


def create_denoiser_and_sampler(
        args: ArgumentParser, 
        input_shape: int,
        input_channels: int,
        out_channels: int,
        rng: torch.Generator,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        only_denoiser: bool = False,
        only_sampler: bool = False
    ):
    """Get the denoiser and sampler if required"""

    model = get_model(args, out_channels, rng, device)

    diffusion_scheme = get_diffusion_scheme(args, device)
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

    sampler = None

    if only_denoiser:
        return denoising_model
    
    elif only_sampler:
        return sampler
    
    return denoising_model, sampler


# ***************************
# Get Callback Method
# ***************************

def create_callbacks(args: ArgumentParser, save_dir: str) -> Sequence[Callback]:
    """Get the callback methods like profilers, metric collectors, etc."""
    
    callbacks = [
        TqdmProgressBar(
            total_train_steps=args.num_train_steps,
            train_monitors=("train_loss",),
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