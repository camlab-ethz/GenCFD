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

"""
File contains all the Base Datasets for Training and Evaluation, as well as datasets with 
macro and micro perturbations for conditional evaluation or fine-tuning. These datasets 
are used for out-of-distribution predictions, with fine-tuning typically applied with 
macro perturbations. Additional utility functions for dataset management are included.

For more information about the individual datasets, refer to fluid_flows_3d.py.
"""

import os
import netCDF4
import numpy as np
import torch
import torch.distributed as dist
import shutil
from typing import Union, Tuple, Any, Dict, List
from torch.distributed import broadcast_object_list, is_initialized
from torch.utils.data import Subset, Dataset

array = np.ndarray
Tensor = torch.Tensor


def train_test_split(dataset: Dataset, split_ratio: float = 0.999, seed: int = 42):
    """
    Split a dataset into training and testing subsets.

    Args:
        dataset: The dataset to split.
        split_ratio: Proportion of the dataset to use for training (default: 0.8).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        A tuple (train_dataset, test_dataset) as subsets of the input dataset.
    """
    num_samples = len(dataset)
    indices = np.arange(num_samples)

    # Shuffle indices deterministically
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    split_idx = int(num_samples * split_ratio)

    train_indices = indices[:split_idx]
    eval_indices = indices[split_idx:]

    train_subset = Subset(dataset, train_indices)
    eval_subset = Subset(dataset, eval_indices)

    return train_subset, eval_subset


class TrainingSetBase:
    """
    Base class for loading and processing datasets related to incompressible fluid flows.
    More details for specific datasets can be found in dataloader/fluid_flows_3d.py

    Args:
        file_system (dict): Contains the file system configuration (paths, etc.). See example in dataloader/fluid_flows_3d.py
        ndim (int): Dimensionality of the dataset (e.g., 2D or 3D).
        input_channel (int): The number of input channels.
        output_channel (int): The number of output channels.
        spatial_resolution (Tuple): Spatial resolution of the data (e.g., grid size).
        input_shape (Tuple): Shape of the input data.
        output_shape (Tuple): Shape of the output data.
        variable_names (List[str]): Names of the variables in the dataset.
        start (int): The starting index (default is 0).
        training_samples (int, optional): Number of training samples (defaults to the total available).
        move_to_local_scratch (bool, optional): Flag to move data to local scratch for fast access (default is False).
        retrieve_stats_from_file (bool, optional): Flag to retrieve statistics (mean and std) from files (default is False).
        mean_training_input (np.ndarray, optional): Pre-calculated mean for input normalization.
        std_training_input (np.ndarray, optional): Pre-calculated std for input normalization.
        mean_training_output (np.ndarray, optional): Pre-calculated mean for output normalization.
        std_training_output (np.ndarray, optional): Pre-calculated std for output normalization.
        get_values (bool, optional): Flag to directly get values without calculation (default is False).
    """

    def __init__(
        self,
        file_system: dict,
        ndim: int,
        input_channel: int,
        output_channel: int,
        spatial_resolution: Tuple,
        input_shape: Tuple,
        output_shape: Tuple,
        variable_names: List[str],
        start: int = 0,
        training_samples: int = None,
        move_to_local_scratch: bool = False,
        retrieve_stats_from_file: bool = False,
        mean_training_input: array = None,
        std_training_input: array = None,
        mean_training_output: array = None,
        std_training_output: array = None,
        get_values: bool = False,
    ) -> None:

        self.start = start
        self.rand_gen = np.random.RandomState(seed=4)

        self.file_system = file_system
        self.input_channel = input_channel
        self.output_channel = output_channel

        self.spatial_resolution = spatial_resolution
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.variable_names = variable_names

        if move_to_local_scratch:
            # Copy file to local scratch
            file_path = self._move_to_local_scratch(
                file_system=file_system, scratch_dir="TMPDIR"
            )
        else:
            # Get the correct file_path and check whether file is on local scratch
            file_path = self.file_on_local_scratch(
                file_system["file_name"], file_system["origin"]
            )

        # Load Dataset
        self.file = netCDF4.Dataset(file_path, "r")

        self.training_samples = (
            self.file.dimensions["member"].size
            if training_samples is None
            else training_samples
        )

        self.mean_training_input = mean_training_input
        self.std_training_input = std_training_input
        self.mean_training_output = mean_training_output
        self.std_training_output = std_training_output

        if retrieve_stats_from_file:
            # Fill mean and std values for the input and output
            self.retrieve_stats_from_file(
                file_system=file_system, ndim=ndim, get_values=get_values
            )

    def __len__(self) -> int:
        return self.training_samples

    def _move_to_local_scratch(self, file_system: dict, scratch_dir: str) -> str:
        """Copy the specified file to the local scratch directory if needed."""

        # Construct the source file path
        data_dir = os.path.join(file_system["origin"], file_system["file_name"])
        file = file_system["file_name"].split("/")[-1]

        # Ensure scratch_dir is correctly resolved
        if scratch_dir == "TMPDIR":
            scratch_dir = os.environ.get(
                "TMPDIR", "/tmp"
            )  # Default to '/tmp' if TMPDIR is undefined

        # Construct the full destination path
        dest_path = os.path.join(scratch_dir, file)

        RANK = int(os.environ.get("LOCAL_RANK", -1))

        # Only copy if the file doesn't exist at the destination
        if not os.path.exists(dest_path) and (RANK == 0 or RANK == -1):
            print(" ")
            print(f"Start copying {file} to {dest_path}...")
            shutil.copy(data_dir, dest_path)
            print("Finished data copy.")

        if is_initialized():
            dist.barrier(device_ids=[RANK])

        return dest_path

    def file_on_local_scratch(self, file_name: str, origin: str) -> str:
        """Checks whether the file is in the local scratch directory or a default path.

        Args:
            file_name (str): The name of the file or the complete path to it.
            origin (str): Specifies the default storage location of the file

        Returns:
            str: The resolved file path, either from the local scratch or a default directory.

        Raises:
            ValueError: If the file cannot be found in any of the expected locations.
        """

        if not file_name:
            raise ValueError("File name must not be empty.")

        tmpdir_file_path = os.path.join(os.environ.get("TMPDIR", ""), file_name)
        if os.path.exists(tmpdir_file_path):
            print(f"Using file from local scratch: {tmpdir_file_path}")
            return tmpdir_file_path

        default_file_path = os.path.join(origin, file_name)
        if os.path.exists(default_file_path):
            return default_file_path

        if os.path.exists(file_name):
            return file_name

        raise ValueError(f"File not found: {file_name}")

    def retrieve_stats_from_file(
        self, file_system: dict, ndim: int, get_values: bool = False
    ) -> None:
        """Given some stats files, the mean and std for training input and output
        can be retrieved

        Args:
            file_system: dictionary with relevant files for the mean and the data
            ndim: dimensionality of the file (number of channels)
            get_values: in some cases the mean can be extracted directly without calculation
        """

        mean_path = os.path.join(file_system["origin_stats"], file_system["mean_file"])
        std_path = os.path.join(file_system["origin_stats"], file_system["std_file"])

        mean_data = np.load(mean_path)
        std_data = np.load(std_path)

        # First half are stats for t0 varialbes, second half are for t1 varialbes
        num_variables = mean_data.shape[-1] // 2

        # t0: Extract the relevant values for the mean and std set
        mean_training_input = mean_data[..., :num_variables]
        std_training_input = std_data[..., :num_variables]

        # t1: Extract the relevant values for the mean and std set
        mean_training_output = mean_data[..., num_variables:]
        std_training_output = std_data[..., num_variables:]

        # Extract the relevant channels
        mean_training_input = mean_training_input[..., : self.input_channel]
        std_training_input = std_training_input[..., : self.input_channel]
        mean_training_output = mean_training_output[..., : self.output_channel]
        std_training_output = std_training_output[..., : self.output_channel]

        if get_values:
            # Extract values directly
            self.mean_training_input = mean_training_input
            self.std_training_input = std_training_input
            self.mean_training_output = mean_training_output
            self.std_training_output = std_training_output

        else:
            # Compute the resulting tensors
            if ndim == 2:
                stats_axis = (0, 1)
            elif ndim == 3:
                stats_axis = (0, 1, 2)
            else:
                raise ValueError(
                    f"Only 2D or 3D datasets are supported and not {ndim}D"
                )
            self.mean_training_input = mean_training_input.mean(axis=stats_axis)
            self.std_training_input = np.mean(std_training_input**2, stats_axis) ** 0.5
            self.mean_training_output = mean_training_output.mean(axis=stats_axis)
            self.std_training_output = (
                np.mean(std_training_output**2, stats_axis) ** 0.5
            )

    def normalize_input(self, u_: Union[array, Tensor]) -> Union[array, Tensor]:

        if self.mean_training_input is not None:
            mean_training_input = self.mean_training_input
            std_training_input = self.std_training_input
            if isinstance(u_, Tensor):
                mean_training_input = torch.as_tensor(
                    mean_training_input, dtype=u_.dtype, device=u_.device
                )
                std_training_input = torch.as_tensor(
                    std_training_input, dtype=u_.dtype, device=u_.device
                )
            return (u_ - mean_training_input) / (std_training_input + 1e-12)
        else:
            return u_

    def denormalize_input(self, u_: Union[array, Tensor]) -> Union[array, Tensor]:

        if self.mean_training_input is not None:
            mean_training_input = self.mean_training_input
            std_training_input = self.std_training_input
            if isinstance(u_, Tensor):
                mean_training_input = torch.as_tensor(
                    mean_training_input, dtype=u_.dtype, device=u_.device
                )
                std_training_input = torch.as_tensor(
                    std_training_input, dtype=u_.dtype, device=u_.device
                )
            return u_ * (std_training_input + 1e-12) + mean_training_input
        else:
            return u_

    def normalize_output(self, u_: Union[array, Tensor]) -> Union[array, Tensor]:

        if self.mean_training_output is not None:
            mean_training_output = self.mean_training_output
            std_training_output = self.std_training_output
            if isinstance(u_, Tensor):
                mean_training_output = torch.as_tensor(
                    mean_training_output, dtype=u_.dtype, device=u_.device
                )
                std_training_output = torch.as_tensor(
                    std_training_output, dtype=u_.dtype, device=u_.device
                )
            return (u_ - mean_training_output) / (std_training_output + 1e-12)
        else:
            return u_

    def denormalize_output(self, u_: Union[array, Tensor]) -> Union[array, Tensor]:

        if self.mean_training_output is not None:
            mean_training_output = self.mean_training_output
            std_training_output = self.std_training_output
            if isinstance(u_, Tensor):
                mean_training_output = torch.as_tensor(
                    mean_training_output, dtype=u_.dtype, device=u_.device
                )
                std_training_output = torch.as_tensor(
                    std_training_output, dtype=u_.dtype, device=u_.device
                )
            return u_ * (std_training_output + 1e-12) + mean_training_output
        else:
            return u_

    def __getitem__(self, item):
        raise NotImplementedError()

    def get_proc_data(self, data):
        return data

    def collate_tf(self, data):
        return data


"""
CONDITIONAL DATASETS FOR EVALUATION

These datasets incorporate both macro and micro perturbations, making them ideal
for evaluation or fine-tuning. Macro perturbations refer to larger shifts in 
the initial conditions, while micro perturbations are small adjustments at the 
scale of a single grid cell. Fine-tuning is typically performed only with macro 
perturbations to adapt the model to broader shifts in conditions.

The main use case is to test or improve a model's robustness to out-of-distribution
inputs by providing data with perturbations that differ from the model's original training conditions.
"""


class ConditionalBase(TrainingSetBase):
    """
    A class for loading and handling datasets with macro and micro perturbations
    for conditional evaluation or fine-tuning purposes.

    Args:
        training_samples (int): Number of training samples.
        file_system (dict): Dictionary containing file paths and configuration.
        ndim (int): Dimensionality of the data (e.g., 2D, 3D).
        input_channel (int): Number of input channels.
        output_channel (int): Number of output channels.
        spatial_resolution (Tuple): Spatial resolution of the data.
        input_shape (Tuple): Shape of the input data.
        output_shape (Tuple): Shape of the output data.
        variable_names (List[str]): Names of variables in the dataset.
        start (int, optional): Starting index for the dataset (default is 0).
        micro_perturbations (int, optional): Number of micro perturbations (default is 0).
        macro_perturbations (int, optional): Number of macro perturbations (default is 0).
        move_to_local_scratch (bool, optional): Flag to move data to local scratch directory (default is False).
    """

    def __init__(
        self,
        training_samples: int,
        file_system: dict,
        ndim: int,
        input_channel: int,
        output_channel: int,
        spatial_resolution: Tuple,
        input_shape: Tuple,
        output_shape: Tuple,
        variable_names: List[str],
        start: int = 0,
        micro_perturbations: int = 0,
        macro_perturbations: int = 0,
        move_to_local_scratch: bool = False,
    ) -> None:

        super().__init__(
            training_samples=training_samples,
            file_system=file_system,
            ndim=ndim,
            input_channel=input_channel,
            output_channel=output_channel,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names,
            start=start,
            move_to_local_scratch=move_to_local_scratch,
        )

        self.micro_perturbations = micro_perturbations
        self.macro_perturbations = macro_perturbations

    def __len__(self) -> int:
        return self.micro_perturbations * self.macro_perturbations

    def get_macro_index(self, index):
        return index // self.micro_perturbations

    def get_micro_index(self, index):
        return index % self.micro_perturbations
