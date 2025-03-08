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

import numpy as np
import torch
import torch.fft
from typing import Union, Tuple
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from GenCFD.dataloader.dataset import (
    train_test_split,
    TrainingSetBase,
)
from GenCFD.dataloader.fluid_flows_2d import (
    ShearLayer2D,
    CloudShock2D,
    RichtmyerMeshkov2D,
    ConditionalShearLayer2D,
    ConditionalCloudShock2D,
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

Tensor = torch.Tensor
Array = np.ndarray
Container = Union[Array, Tensor]


# ***************************
# Load Dataset and Dataloader
# ***************************


def get_dataset(
    name: str,
    is_time_dependent: bool = False,
    # device: torch.device = None
) -> TrainingSetBase:
    """Returns the correct dataset and if the dataset has a time dependency
    This is necessary for the evaluation pipeline if there is no json file
    provided.
    """

    metadata = METADATA_CLASSES[name]

    if name == "ShearLayer2D":
        dataset = ShearLayer2D(metadata=metadata)
        time_cond = False

    elif name == "CloudShock2D":
        dataset = CloudShock2D(metadata=metadata)
        time_cond = False

    elif name == "RichtmyerMeshkov2D":
        dataset = RichtmyerMeshkov2D(metadata=metadata)
        time_cond = False

    elif name == "ShearLayer3D":
        dataset = ShearLayer3D(metadata=metadata)
        time_cond = True

    elif name == "TaylorGreen3D":
        dataset = TaylorGreen3D(metadata=metadata)
        time_cond = True

    elif name == "Nozzle3D":
        dataset = Nozzle3D(metadata=metadata)
        time_cond = True

    elif name == "ConditionalShearLayer2D":
        dataset = ConditionalShearLayer2D(metadata=metadata)
        time_cond = False

    elif name == "ConditionalCloudShock2D":
        dataset = ConditionalCloudShock2D(metadata=metadata)
        time_cond = False

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
            # shuffle=True if args.world_size == 1 else False,
            shuffle=True,
            pin_memory=True,
            num_workers=num_worker,
            prefetch_factor=prefetch_factor,
            sampler=sampler if args.world_size > 1 else None,
        )
        return (dataloader, dataset, time_cond)


def normalize(
    u_: Container, mean: Container = None, std: Container = None
) -> Container:
    """ "Normalizes data input by subtracting mean and dividing by std.

    Args:
        u_ (Tensor or ndarray): The input data to normalize.
        mean (Tensor or ndarray, optional): Mean values for normalization. Should be of same type as u_.
        std (Tensor or ndarray, optional): Std values for normalization. Should be of same type as u_.

    Returns:
        Tensor or ndarray: Normalized data in same type as input u_.
    """

    if mean is not None and std is not None:
        if isinstance(u_, Tensor) and all(
            isinstance(var, Array) for var in (mean, std)
        ):
            mean = torch.tensor(mean, dtype=u_.dtype, device=u_.device)
            std = torch.tensor(std, dtype=u_.dtype, device=u_.device)
        elif isinstance(u_, Array) and all(
            isinstance(var, Tensor) for var in (mean, std)
        ):
            mean = mean.cpu().numpy()
            std = std.cpu().numpy()
        return (u_ - mean) / (std + 1e-12)
    else:
        return u_


def denormalize(
    u_: Container, mean: Container = None, std: Container = None
) -> Container:
    """Denormalizes data by applying std and mean used for normalization.

    Args:
        u_ (Tensor or ndarray): The normalized data to revert.
        mean (Tensor or ndarray, optional): Mean values used for normalization.
        std (Tensor or ndarray, optional): Std values used for normalization.

    Returns:
        Tensor or ndarray: Denormalized data in the same type as input u_.
    """

    if mean is not None and std is not None:
        if isinstance(u_, Tensor) and all(
            isinstance(var, Array) for var in (mean, std)
        ):
            mean = torch.tensor(mean, dtype=u_.dtype, device=u_.device)
            std = torch.tensor(std, dtype=u_.dtype, device=u_.device)
        elif isinstance(u_, Array) and all(
            isinstance(var, Tensor) for var in (mean, std)
        ):
            mean = mean.cpu().numpy()
            std = std.cpu().numpy()
        return u_ * (std + 1e-12) + mean
    else:
        return u_

