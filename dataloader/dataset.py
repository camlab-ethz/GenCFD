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
File contains all the used datasets. The keyword IC is a characterization of 
incompressible flows. 

    Dataset:
    2D Shear Layer Problem:   DataIC_Vel              ConditionalDataIC_Vel
    2D Cloud Shock:           DataIC_Cloud_Shock_2D   ConditionalDataIC_Cloud_Shock_2D
    3D Shear Layer:           DataIC_3D_Time          ConditionalDataIC_3D
    3D Taylor Green Vortex:   DataIC_3D_Time_TG       ConditionalDataIC_3D_TG
"""

import os
import netCDF4
import numpy as np
import torch
import torch.distributed as dist
import shutil
from typing import Union, Any
from torch.distributed import is_initialized
from torch.utils.data import Subset

DIR_PATH_LOADER = '/cluster/work/math/camlab-data/data/diffusion_project'

array = np.ndarray
Tensor = torch.Tensor


def train_test_split(dataset, split_ratio=0.8, seed=42):
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


def create_file_system(metadata: Any) -> dict:
    """Creates the filesystem with the metadata provided
    Some parts are None values and just placeholders."""

    return {
        'file_name': metadata.file_name,
        'origin': metadata.origin,
        'mean_file': metadata.mean_file,
        'std_file': metadata.std_file,
        'stats_file': metadata.stats_file,
        'origin_stats': metadata.origin_stats,
        'conditional_path': metadata.conditional_path
    }


class TrainingSetBase:
    def __init__(
        self,
        training_samples: int,
        file_system: dict,
        input_channel: int,
        output_channel: int,
        start: int = 0,
    ) -> None:

        self.start = start
        self.training_samples = training_samples
        self.rand_gen = np.random.RandomState(seed = 4)

        self.file_system = file_system
        self.input_channel = input_channel
        self.output_channel =  output_channel
    
        self.mean_training_input = None
        self.std_training_input = None
        self.mean_training_output = None
        self.std_training_output = None

    def __len__(self):
        return self.training_samples

    def _move_to_local_scratch(self, file_system: dict, scratch_dir: str) -> str:
        """Copy the specified file to the local scratch directory if needed."""
        
        # Construct the source file path
        data_dir = os.path.join(file_system['origin'], file_system['file_name'])
        file = file_system['file_name'].split("/")[-1]
        
        # Ensure scratch_dir is correctly resolved
        if scratch_dir == 'TMPDIR':
            scratch_dir = os.environ.get('TMPDIR', '/tmp')  # Default to '/tmp' if TMPDIR is undefined
        
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
        
        tmpdir_file_path = os.path.join(os.environ.get('TMPDIR', ''), file_name)
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
            self, 
            file_system: dict, 
            ndim: int,
            get_values: bool = False
        ) -> None:
        """Given some stats files, the mean and std for training input and output 
        can be retrieved
        
        Args:
            file_system: dictionary with relevant files for the mean and the data
            ndim: dimensionality of the file (number of channels)
            get_values: in some cases the mean can be extracted directly without calculation
        """
        
        mean_path = os.path.join(file_system['origin_stats'], file_system['mean_file'])
        std_path = os.path.join(file_system['origin_stats'], file_system['std_file'])

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
        mean_training_input = mean_training_input[..., :self.input_channel]
        std_training_input = std_training_input[..., :self.input_channel]
        mean_training_output = mean_training_output[..., :self.output_channel]
        std_training_output = std_training_output[..., :self.output_channel]

        if get_values:
            # Exract values directly
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
                raise ValueError(f"Only 2D or 3D datasets are supported and not {ndim}D")
            self.mean_training_input = mean_training_input.mean(axis=stats_axis)
            self.std_training_input = np.mean(std_training_input ** 2, stats_axis) ** 0.5
            self.mean_training_output = mean_training_output.mean(axis=stats_axis)
            self.std_training_output = np.mean(std_training_output ** 2, stats_axis) ** 0.5


    def normalize_input(self, u_: Union[array, Tensor]) -> Union[array, Tensor]:

        if self.mean_training_input is not None:
            mean_training_input = self.mean_training_input
            std_training_input = self.std_training_input
            if isinstance(u_, Tensor):
                mean_training_input = torch.as_tensor(mean_training_input, dtype=u_.dtype, device=u_.device)
                std_training_input = torch.as_tensor(std_training_input, dtype=u_.dtype, device=u_.device)
            return (u_ - mean_training_input) / (std_training_input + 1e-12)
        else:
            return u_

    def denormalize_input(self, u_: Union[array, Tensor]) -> Union[array, Tensor]:
        
        if self.mean_training_input is not None:
            mean_training_input = self.mean_training_input
            std_training_input = self.std_training_input
            if isinstance(u_, Tensor):
                mean_training_input = torch.as_tensor(mean_training_input, dtype=u_.dtype, device=u_.device)
                std_training_input = torch.as_tensor(std_training_input, dtype=u_.dtype, device=u_.device)
            return u_ * (std_training_input + 1e-12) + mean_training_input
        else:
            return u_

    def normalize_output(self, u_: Union[array, Tensor]) -> Union[array, Tensor]:

        if self.mean_training_output is not None:
            mean_training_output = self.mean_training_output
            std_training_output = self.std_training_output
            if isinstance(u_, Tensor):
                mean_training_output = torch.as_tensor(mean_training_output, dtype=u_.dtype, device=u_.device)
                std_training_output = torch.as_tensor(std_training_output, dtype=u_.dtype, device=u_.device)
            return (u_ - mean_training_output) / (std_training_output + 1e-12)
        else:
            return u_

    def denormalize_output(self, u_: Union[array, Tensor]) -> Union[array, Tensor]:
        
        if self.mean_training_output is not None:
            mean_training_output = self.mean_training_output
            std_training_output = self.std_training_output
            if isinstance(u_, Tensor):
                mean_training_output = torch.as_tensor(mean_training_output, dtype=u_.dtype, device=u_.device)
                std_training_output = torch.as_tensor(std_training_output, dtype=u_.dtype, device=u_.device)
            return u_ * (std_training_output + 1e-12) + mean_training_output
        else:
            return u_

    def __getitem__(self, item):
        raise NotImplementedError()

    def get_proc_data(self, data):
        return data

    def collate_tf(self, data):
        return data


#################### IC TO HIGH RESOLUTION ####################

class DataIC_Vel(TrainingSetBase):
    def __init__(
        self, 
        metadata: Any,
        start: int = 0,
        file: str = None,
        move_to_local_scratch: bool = True,
    ):
        
        self.class_name = self.__class__.__name__

        input_channel = 2
        output_channel = 2
        self.spatial_resolution = (128, 128)
        self.input_shape = (4, 128, 128)
        self.output_shape = (2, 128, 128)
    
        self.metadata = metadata
        file_system = create_file_system(metadata)

        if move_to_local_scratch:
            # Copy file to local scratch
            file_path = self._move_to_local_scratch(file_system=file_system, scratch_dir='TMPDIR')
        else:
            # Get the correct file_path and check whether file is on local scratch
            file_path = self.file_on_local_scratch(file_system['file_name'], file_system['origin'])

        self.file = netCDF4.Dataset(file_path, 'r')

        super().__init__(
            start = start, 
            training_samples=self.file['data'].shape[0], 
            file_system=file_system,
            input_channel=input_channel, 
            output_channel=output_channel,
        )

        # Set mean and std for training input and output
        self.retrieve_stats_from_file(file_system=file_system, ndim=2)


    def __getitem__(self, index):

        index += self.start        
        data = self.file['data'][index].data

        data_input = data[0, ..., :self.input_channel]
        data_output = data[1, ..., :self.output_channel]

        data_input = self.normalize_input(data_input)
        data_output = self.normalize_output(data_output)

        initial_cond = (
            torch.from_numpy(data_input)
            .type(torch.float32)
            .permute(2, 1, 0)
            .cpu()
        )

        target_cond = (
            torch.from_numpy(data_output)
            .type(torch.float32)
            .permute(2, 1, 0)
            .cpu()
        )

        return {
            'initial_cond': initial_cond,
            'target_cond': target_cond
        }
        

#################### 2D CLOUD SHOCK DATASETS ####################

class DataIC_Cloud_Shock_2D(TrainingSetBase):
    def __init__(
            self, 
            metadata: Any,
            start = 0, 
            file = None,
            move_to_local_scratch: bool = True
        ):

        self.class_name = self.__class__.__name__
        input_channel = 4
        output_channel = 4
        self.spatial_resolution = (128, 128)
        self.input_shape = (8, 128, 128)
        self.output_shape = (4, 128, 128)

        self.metadata = metadata
        file_system = create_file_system(metadata)

        if move_to_local_scratch:
            # Copy file to local scratch
            file_path = self._move_to_local_scratch(file_system=file_system, scratch_dir='TMPDIR')
        else:
            # Get the correct file_path and check whether file is on local scratch
            file_path = self.file_on_local_scratch(file_system['file_name'], file_system['origin'])

        self.file = netCDF4.Dataset(file_path, 'r')

        super().__init__(
            training_samples=self.file['data'].shape[0],
            file_system=file_system,
            input_channel=input_channel,
            output_channel=output_channel,
            start=start
        )

        self.retrieve_stats_from_file(file_system=file_system, ndim=2)

    def __getitem__(self, index):

        index += self.start        
        data = self.file['data'][index].data

        data_input = data[0, ..., :self.input_channel]
        data_output = data[1, ..., :self.output_channel]

        data_input = self.normalize_input(data_input)
        data_output = self.normalize_output(data_output)

        initial_cond = (
            torch.from_numpy(data_input)
            .type(torch.float32)
            .permute(2, 1, 0)
        )

        target_cond = (
            torch.from_numpy(data_output)
            .type(torch.float32)
            .permute(2, 1, 0)
        )

        return {
            'initial_cond': initial_cond,
            'target_cond': target_cond
        }

#################### 2D RICHTMYER-MESHKOV DATASETS ####################

class RichtmyerMeshkov2D(TrainingSetBase):
    def __init__(
            self, 
            metadata: Any,
            start = 0, 
            file = None,
            move_to_local_scratch: bool = True
        ):

        self.class_name = self.__class__.__name__

        # Channels: ['rho', 'E', 'mx', 'my']
        input_channel = 4
        output_channel = 4
        self.spatial_resolution = (128, 128)
        self.input_shape = (8, 128, 128)
        self.output_shape = (4, 128, 128)

        self.metadata = metadata
        file_system = create_file_system(metadata)

        if move_to_local_scratch:
            # Copy file to local scratch
            file_path = self._move_to_local_scratch(file_system=file_system, scratch_dir='TMPDIR')
        else:
            # Get the correct file_path and check whether file is on local scratch
            file_path = self.file_on_local_scratch(file_system['file_name'], file_system['origin'])

        self.file = netCDF4.Dataset(file_path, 'r')

        super().__init__(
            training_samples=self.file.dimensions['member'].size,
            file_system=file_system,
            input_channel=input_channel,
            output_channel=output_channel,
            start=start
        )

        self.retrieve_stats_from_file(file_system=file_system, ndim=2, get_values=True)

    def __getitem__(self, index):

        index += self.start  
  
        # Load the data for the given index
        rho_data = self.file.variables['rho'][index]  # Shape: (2, 128, 128)
        E_data = self.file.variables['E'][index]
        mx_data = self.file.variables['mx'][index]
        my_data = self.file.variables['my'][index]

        # Stack along the new last dimension (axis=-1)
        combined_data = np.stack((rho_data, E_data, mx_data, my_data), axis=-1)  # Shape: (2, 128, 128, 4)

        # Extract initial and final conditions
        initial_condition = self.normalize_input(
            combined_data[0, ...])  # Shape: (128, 128, 4)
        target_condition = self.normalize_output(
            combined_data[1, ...])  # Shape: (128, 128, 4)

        initial_cond = (
            torch.from_numpy(initial_condition)
            .type(torch.float32)
            .permute(2, 1, 0)
        )

        target_cond = (
            torch.from_numpy(target_condition)
            .type(torch.float32)
            .permute(2, 1, 0)
        )

        return {
            'initial_cond': initial_cond,
            'target_cond': target_cond
        }

#################### 3D SHEAR LAYER DATASETS ####################

class DataIC_3D_Time(TrainingSetBase):
    def __init__(
            self,
            metadata: Any,
            start=0,
            file = None,
            move_to_local_scratch: bool = True
        ):

        input_channel = 3
        output_channel = 3
        self.spatial_resolution = (64, 64, 64)
        self.input_shape = (6, 64, 64, 64)
        self.output_shape = (3, 64, 64, 64)

        self.metadata = metadata
        file_system = create_file_system(metadata)

        if move_to_local_scratch:
            # Copy file to local scratch
            file_path = self._move_to_local_scratch(file_system=file_system, scratch_dir='TMPDIR')
        else:
            # Get the correct file_path and check whether file is on local scratch
            file_path = self.file_on_local_scratch(file_system['file_name'], file_system['origin'])

        self.file = netCDF4.Dataset(file_path, 'r')

        super().__init__( 
            training_samples=self.file.variables['data'].shape[0],
            file_system=file_system,
            input_channel=input_channel,
            output_channel=output_channel,
            start=start
        )

        self.n_all_steps = 10
        self.start = self.start // self.n_all_steps

        # Hardcoded values for the 3D Shear Flow Vertex Dataset
        self.mean_training_input = np.array([1.5445266e-08, 1.2003070e-08, -3.2182508e-09])
        self.mean_training_output = np.array([-8.0223117e-09, -3.3674191e-08, 1.5241447e-08])

        self.std_training_input = np.array([0.20691067, 0.15985465, 0.15808222])
        self.std_training_output = np.array([0.2706984, 0.24893111, 0.24169469])


    def __getitem__(self, index):
        index += self.start
        data = self.file.variables['data'][index].data
        if self.start == 0:
            lead_time = self.file.variables['lead_time'][index].data
        else:
            lead_time = 1.

        data_input = self.normalize_input(data[0])
        data_output = self.normalize_output(data[1])

        initial_cond = (
            torch.from_numpy(data_input)
            .type(torch.float32)
            .permute(3, 2, 1, 0)
        )
        target_cond = (
            torch.from_numpy(data_output)
            .type(torch.float32)
            .permute(3, 2, 1, 0)
        )

        return {
            'lead_time': torch.tensor(lead_time, dtype=torch.float32), 
            'initial_cond': initial_cond,
            'target_cond': target_cond
        }


#################### 3D TAYLOR GREEN VORTEX DATASET ####################

class DataIC_3D_Time_TG(TrainingSetBase):
    
    def __init__(
            self,
            metadata: Any,
            start: int = 0,
            file: str = None,
            min_time: int = 0,
            max_time: int = 5,
            move_to_local_scratch: bool = True
    ):

        input_channel = 3
        output_channel = 3

        self.metadata = metadata
        file_system = create_file_system(metadata)

        if move_to_local_scratch:
            # Copy file to local scratch
            file_path = self._move_to_local_scratch(file_system=file_system, scratch_dir='TMPDIR')
        else:
            # Get the correct file_path and check whether file is on local scratch
            file_path = self.file_on_local_scratch(file_system['file_name'], file_system['origin'])

        self.file = netCDF4.Dataset(file_path, 'r')

        super().__init__(
            training_samples=self.file.variables['u'].shape[0],
            file_system=file_system,
            input_channel=input_channel,
            output_channel=output_channel,
            start=start,
        )

        self.retrieve_stats_from_file(file_system=file_system, ndim=3)

        self.spatial_resolution = (64, 64, 64)
        self.input_shape = (6, 64, 64, 64)
        self.output_shape = (3, 64, 64, 64)

        self.min_time = min_time
        self.max_time = max_time

        # Precompute all possible (t_initial, t_final) pairs within the specified range.
        self.time_pairs = [(i, j) for i in range(self.min_time, self.max_time) for j in range(i + 1, self.max_time + 1)]
        self.total_pairs = len(self.time_pairs)

        
    def __len__(self):
        # Return the total number of data points times the number of pairs.
        return len(self.file.variables['u']) * self.total_pairs


    def __getitem__(self, index):
        # Determine the data point and the (t_initial, t_final) pair
        data_index = index // self.total_pairs
        pair_index = index % self.total_pairs
        t_initial, t_final = self.time_pairs[pair_index]

        # Load the data for the given index
        u_data = self.file.variables['u'][data_index]  # Shape: (6, 64, 64, 64)
        v_data = self.file.variables['v'][data_index]
        w_data = self.file.variables['w'][data_index]

        # Stack along the new last dimension (axis=-1)
        combined_data = np.stack((u_data, v_data, w_data), axis=-1)  # Shape: (6, 64, 64, 64, 3)

        # Extract initial and final conditions
        initial_condition = self.normalize_input(
            combined_data[t_initial])  # Shape: (64, 64, 64, 3)
        final_condition = self.normalize_output(
            combined_data[t_final])  # Shape: (64, 64, 64, 3)
        
        # Linearly remap the lead_time in the interval [0.25, 2.0].
        lead_time = float(t_final - t_initial)
        lead_time_normalized = 0.25 + 0.4375 * (lead_time - 1)

        initial_cond = (
            torch.from_numpy(initial_condition)
            .type(torch.float32)
            .permute(3, 2, 1, 0)
        )

        target_cond = (
            torch.from_numpy(final_condition)
            .type(torch.float32)
            .permute(3, 2, 1, 0)
        )

        return {
            'lead_time': torch.tensor(lead_time_normalized, dtype=torch.float32), 
            'initial_cond': initial_cond,
            'target_cond': target_cond
        }


#################### CONDITIONAL DATASETS FOR EVALUATON ####################
"""
These datasets have some macro and micro perturbations incorporated. Those
can be used for evaluation or fine tuning. This allows for a out of distribution
prediction. Finetuning is typically done only with macro perturbations.

Macro perturbations are bigger shifts in the initial conditions, while micro
perturbations are in the area of 1 grid cell.
"""

class ConditionalBase(TrainingSetBase):

    def __init__(
            self,
            training_samples: int,
            file_system: dict,
            input_channel: int,
            output_channel: int,
            start: int = 0,
            micro_perturbation: int = 0,
            macro_perturbation: int = 0,
            file_path: str = None,
            stat_folder: str = None,
            t_final: int = None
        ) -> None : 

        super().__init__(
            training_samples=training_samples, 
            file_system=file_system,
            input_channel=input_channel,
            output_channel=output_channel,
            start=start
        )

        self.micro_perturbation = micro_perturbation
        self.macro_perturbation = macro_perturbation
        self.file_path = file_path
        self.stat_folder = stat_folder
        self.t_final = t_final
        self.mean_down, self.std_down, self.kk, self.spectrum, self.energy, self.idx_wass, self.sol_wass = None, None, None, None, None, None, None

    def set_true_stats_for_macro(self, macro_idx: int):

        '''
            macro_idx : macro index to be tested on
    
            There must exist a folder self.stat_folder with files Stats_{macro_idx}.nc in it.
            Each Stats_{macro_idx}.nc file must have the following variables:
            
              - mean_     : target mean of the distribution
              - std_      : target std of the distribution
              - spectrum_ : target spectra of each variable
              - energy_   : target energy of each variable
              - idx       : locations at which we compute wasserstein distance
                            -- while computing statistics:
                            -- ns  = 1000
                            -- ids = np.random.choice(size ** spatial_dim, min(ns, size ** spatial_dim), replace=False)
              - sol_      : values of the samples at idx points
        '''
        
        assert self.stat_folder is not None
        
        file_name = f"{self.stat_folder}/Stats_{macro_idx}.nc"
        if not os.path.isfile(file_name):
            raise FileNotFoundError(f"Statistics file {file_name} not found. Run ComputeTrueStatistics.py")
        
        f = netCDF4.Dataset(file_name, 'r')
        mean = np.array(f.variables['mean_'][:])
        std = np.array(f.variables['std_'][:])
        spectrum = np.array(f.variables['spectrum_'][:])
        energy = np.array(f.variables['energy_'][:])
        idx_wass = np.array(f.variables['idx'][:])
        sol_wass = np.array(f.variables['sol_'][:])
        kk = np.arange(1, spectrum.shape[0] + 1)

        self.mean_down, self.std_down, self.kk, self.spectrum, self.energy, self.idx_wass, self.sol_wass = mean, std, kk, spectrum, energy, idx_wass, sol_wass

    def len(self):
        return self.micro_perturbation * self.macro_perturbation

    def get_macro_index(self, index):
        return index // self.micro_perturbation

    def get_micro_index(self, index):
        return index % self.micro_perturbation

#--------------------------------------

class ConditionalDataIC_Vel(ConditionalBase):
    def __init__(
            self, 
            metadata: Any,
            move_to_local_scratch: bool = True
        ):
        
        input_channel = 2
        output_channel = 2

        self.spatial_resolution = (128, 128)
        self.input_shape = (4, 128, 128)
        self.output_shape = (2, 128, 128)

        self.metadata = metadata
        file_system = create_file_system(metadata)

        if move_to_local_scratch:
            # Copy file to local scratch
            file_path = self._move_to_local_scratch(file_system=file_system, scratch_dir='TMPDIR')
        else:
            # Get the correct file_path and check whether file is on local scratch
            file_path = self.file_on_local_scratch(file_system['file_name'], file_system['origin'])

        stats_path = self.file_on_local_scratch(file_system['stats_file'], file_system['origin_stats'])
        self.file = netCDF4.Dataset(file_path, mode='r')

        self.start_index = 0    

        samples_shape = self.file.variables['data'].shape
        training_samples = samples_shape[0] * samples_shape[1] # macro * micro perturbations

        super().__init__(
            training_samples=training_samples,
            file_system=file_system,
            input_channel=input_channel,
            output_channel=output_channel,
            start=self.start_index,
            micro_perturbation = 1000,
            macro_perturbation = 10,
            file_path = file_path,
            stat_folder = stats_path)  
         
        # mean_training_input = np.array([8.0606696e-08, 4.8213877e-11])
        # std_training_input = np.array([0.19003302, 0.13649726])
        # mean_training_output = np.array([4.9476512e-09, -1.5097612e-10])
        # std_training_output = np.array([0.35681796, 0.5053845]) 


    def __getitem__(self, index):
        macro_idx = self.get_macro_index(index + self.start_index)
        micro_idx = self.get_micro_index(index + self.start_index)
        datum = self.file.variables['data'][macro_idx, micro_idx].data

        data_input = datum[0, ..., :self.input_channel]
        data_output = datum[-1, ..., :self.output_channel]

        initial_cond = (
            torch.from_numpy(data_input)
            .type(torch.float32)
            .permute(2, 1, 0)
        )
        target_cond = (
            torch.from_numpy(data_output)
            .type(torch.float32)
            .permute(2, 1, 0)
        )
        # Store indices for the CDF computation
        return {
            'initial_cond': initial_cond,
            'target_cond': target_cond,
            # 'sample_idx': index,
            # 'macro_idx': macro_idx,
            # 'micro_idx': micro_idx
        }


class ConditionalDataIC_Cloud_Shock_2D(ConditionalBase):
    def __init__(
            self, 
            metadata: Any,
            move_to_local_scratch: bool = True
        ):
        
        input_channel = 4
        output_channel = 4

        self.spatial_resolution = (128, 128)
        self.input_shape = (8, 128, 128)
        self.output_shape = (4, 128, 128)
        
        self.metadata = metadata
        file_system = create_file_system(metadata)

        if move_to_local_scratch:
            # Copy file to local scratch
            file_path = self._move_to_local_scratch(file_system=file_system, scratch_dir='TMPDIR')
        else:
            # Get the correct file_path and check whether file is on local scratch
            file_path = self.file_on_local_scratch(file_system['file_name'], file_system['origin'])

        self.file = netCDF4.Dataset(file_path, mode='r')
        self.start_index = 0

        samples_shape = self.file.variables['data'].shape
        training_samples = samples_shape[0] * samples_shape[1] # macro * micro perturbations

        super().__init__(
            training_samples=training_samples,
            file_system=file_system,
            input_channel=input_channel,
            output_channel=output_channel,
            start=self.start_index,
            micro_perturbation = 1000,
            macro_perturbation = 10,
            file_path = file_path)  
    

    def __getitem__(self, index):
        macro_idx = self.get_macro_index(index + self.start_index)
        micro_idx = self.get_micro_index(index + self.start_index)
        datum = self.file.variables['data'][macro_idx, micro_idx].data

        data_input = datum[0, ..., :self.input_channel]
        data_output = datum[-1, ..., :self.output_channel]

        initial_cond = (
            torch.from_numpy(data_input)
            .type(torch.float32)
            .permute(2, 1, 0)
        )
        target_cond = (
            torch.from_numpy(data_output)
            .type(torch.float32)
            .permute(2, 1, 0)
        )
        
        # Store indices for the CDF computation
        return {
            'initial_cond': initial_cond,
            'target_cond': target_cond,
            # 'sample_idx': index,
            # 'macro_idx': macro_idx,
            # 'micro_idx': micro_idx
        }


class ConditionalDataIC_3D(ConditionalBase):
    def __init__(
            self, 
            metadata: Any,
            move_to_local_scratch: bool = True
        ):

        input_channel = 3
        output_channel = 3

        self.spatial_resolution = (64, 64, 64)
        self.input_shape = (6, 64, 64, 64)
        self.output_shape = (3, 64, 64, 64)

        self.macro_perturbation = 10
        self.micro_perturbation = 1000

        self.time_step = -1

        self.metadata = metadata
        file_system = create_file_system(metadata)

        if move_to_local_scratch:
            # Copy file to local scratch
            file_path = self._move_to_local_scratch(file_system=file_system, scratch_dir='TMPDIR')
        else:
            # Get the correct file_path and check whether file is on local scratch
            file_path = self.file_on_local_scratch(file_system['file_name'], file_system['origin'])

        self.file = netCDF4.Dataset(file_path, mode='r')

        # shape of the dataset: (macro, micro, time, x, y, z, c)
        data_shape = self.file['data'].shape
        training_samples = data_shape[0] * data_shape[1]
        micro_perturbations = data_shape[1]
        macro_perturbations = data_shape[0]

        super().__init__(
            training_samples=training_samples,
            file_system=file_system,
            input_channel=input_channel,
            output_channel=output_channel,
            micro_perturbation=micro_perturbations,
            macro_perturbation=macro_perturbations,
            file_path=file_path,
        )

        # Set the same mean and std values as for training DataIC_3D_Time
        # self.mean_training_input = np.array([1.5445266e-08, 1.2003070e-08, -3.2182508e-09])
        # self.mean_training_output = np.array([-8.0223117e-09, -3.3674191e-08, 1.5241447e-08])

        # self.std_training_input = np.array([0.20691067, 0.15985465, 0.15808222])
        # self.std_training_output = np.array([0.2706984, 0.24893111, 0.24169469])


    def __getitem__(self, index):
        macro_idx = self.get_macro_index(index)
        micro_idx = self.get_micro_index(index)
        datum = self.file.variables['data'][macro_idx, micro_idx, (0, self.time_step)].data

        data_input = datum[0]
        data_output = datum[-1]

        initial_cond = (
            torch.from_numpy(data_input)
            .type(torch.float32)
            .permute(3, 2, 1, 0)
        )

        target_cond = (
            torch.from_numpy(data_output)
            .type(torch.float32)
            .permute(3, 2, 1, 0)
        )

        # Store indices for the CDF computation
        return {
            # lead_time needs to be changed depending on your start index
            'lead_time': torch.tensor(1., dtype=torch.float32),
            'initial_cond': initial_cond,
            'target_cond': target_cond,
            # 'sample_idx': index,
            # 'macro_idx': macro_idx,
            # 'micro_idx': micro_idx
        }


class ConditionalDataIC_3D_TG(ConditionalBase):
    def __init__(
            self, 
            metadata: Any,
            t_final: int = 5, 
            move_to_local_scratch: bool = True):

        input_channel = 3
        output_channel = 3

        self.spatial_resolution = (64, 64, 64)
        self.input_shape = (6, 64, 64, 64)
        self.output_shape = (3, 64, 64, 64)

        self.t_final = t_final

        self.start = 0
        self.start_index = 0
        
        self.metadata = metadata
        file_system = create_file_system(metadata)
        
        if move_to_local_scratch:
            # Copy file to local scratch
            file_path = self._move_to_local_scratch(file_system=file_system, scratch_dir='TMPDIR')
        else:
            # Get the correct file_path and check whether file is on local scratch
            file_path = self.file_on_local_scratch(file_system['file_name'], file_system['origin'])

        self.file = netCDF4.Dataset(file_path, mode='r')

        u_shape = self.file.variables['u'].shape  # Shape: (6, 64, 64, 64)
        v_shape = self.file.variables['v'].shape
        w_shape = self.file.variables['w'].shape
        assert (u_shape == v_shape == w_shape), "Data needs to align in terms of shape!"
        data_shape = u_shape

        training_samples = data_shape[0] * data_shape[1]
        micro_perturbations = data_shape[1]
        macro_perturbations = data_shape[0]

        super().__init__(
            training_samples=training_samples,
            file_system=file_system,
            input_channel=input_channel,
            output_channel=output_channel,
            start=self.start,
            micro_perturbation=micro_perturbations,
            macro_perturbation=macro_perturbations,
            file_path=file_path,
            # stat_folder=stat_folder,
            t_final=self.t_final
        )
    
    def __getitem__(self, index):
        macro_idx = self.get_macro_index(index + self.start_index)
        micro_idx = self.get_micro_index(index + self.start_index)

        # Shape: (64, 64, 64)
        u_data = self.file.variables['u'][macro_idx, micro_idx, 0]
        v_data = self.file.variables['v'][macro_idx, micro_idx, 0]
        w_data = self.file.variables['w'][macro_idx, micro_idx, 0]

        # Stack along the new last dimension (axis=-1), Shape: (64, 64, 64, 3)
        data_input = np.stack((u_data, v_data, w_data), axis=-1) 

        # Load the data for the given index, Shape: (64, 64, 64)
        u_data = self.file.variables['u'][macro_idx, micro_idx, self.t_final]
        v_data = self.file.variables['v'][macro_idx, micro_idx, self.t_final]
        w_data = self.file.variables['w'][macro_idx, micro_idx, self.t_final]
        # Shape: (64, 64, 64, 3)
        data_output = np.stack((u_data, v_data, w_data), axis=-1) 

        initial_cond = (
            torch.from_numpy(data_input)
            .type(torch.float32)
            .permute(3, 2, 1, 0)
        )

        target_cond = (
            torch.from_numpy(data_output)
            .type(torch.float32)
            .permute(3, 2, 1, 0)
        )

        lead_time = float(self.t_final - self.start_index)
        # lead_time goes from 0 to 2
        lead_time_normalized = 0.25 + 0.4375 * (lead_time - 1)
        
        # Store indices for the CDF computation
        return {
            'lead_time': torch.tensor(lead_time_normalized, dtype=torch.float32),
            'initial_cond': initial_cond,
            'target_cond': target_cond,
            # 'sample_idx': index,
            # 'macro_idx': macro_idx,
            # 'micro_idx': micro_idx
        }