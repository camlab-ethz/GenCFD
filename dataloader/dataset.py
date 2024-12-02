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

List of used Datasets:
- 2D Incompressible Flow
- 3D Shear Layer
- 3D Taylor Green Vortex
"""

import os
import netCDF4
import numpy as np
import torch
from typing import Union

DIR_PATH_LOADER = '/cluster/work/math/camlab-data/data/diffusion_project'

array = np.ndarray
Tensor = torch.Tensor


class TrainingSetBase:
    def __init__(self,
                 training_samples: int,
                 start: int = 0,
    ) -> None:
        
        self.start = start
        self.training_samples = training_samples
        self.rand_gen = np.random.RandomState(seed = 4)
    
        self.mean_training_input = None
        self.std_training_input = None
        self.mean_training_output = None
        self.std_training_output = None

    def __len__(self):
        return self.training_samples

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
    def __init__(self, 
                 start: int = 0,
                 file: str = None):
        
        self.class_name = self.__class__.__name__
        self.input_channel = 2
        self.output_channel = 2
        self.spatial_resolution = (128, 128)
        self.input_shape = (4, 128, 128)
        self.output_shape = (2, 128, 128)
        
        
        if file is None:
            self.file = netCDF4.Dataset(f'{DIR_PATH_LOADER}/ddsl_fast_nothing_128_tr2.nc', mode='r')
        else:
            self.file = file

        super().__init__(start = start, training_samples=self.file['data'].shape[0])
        
        self.mean_training_input = np.array([8.0606696e-08, 4.8213877e-11])
        self.std_training_input = np.array([0.19003302, 0.13649726])
        self.mean_training_output = np.array([4.9476512e-09, -1.5097612e-10])
        self.std_training_output = np.array([0.35681796, 0.5053845])

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
    
#################### 3D SHEAR LAYER DATASETS ####################

class DataIC_3D_Time(TrainingSetBase):
    def __init__(
            self,
            start=0,
            file = None
        ):

        self.input_channel = 3
        self.output_channel = 3
        self.spatial_resolution = (64, 64, 64)
        self.input_shape = (6, 64, 64, 64)
        self.output_shape = (3, 64, 64, 64)

        if file is None:
            if start == 0:
                self.file_path = '/cluster/work/math/camlab-data/data/diffusion_project/shear_layer_3D_64_all_time_2.nc'
            else:
                self.file_path = '/cluster/work/math/camlab-data/data/diffusion_project/shear_layer_3D_64_smaller.nc'

        self.file = netCDF4.Dataset(self.file_path, 'r')

        super().__init__(start=start, training_samples=self.file.variables['data'].shape[0])

        self.n_all_steps = 10
        self.start = self.start // self.n_all_steps

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
            start: int = 0,
            file: str = None,
            min_time: int = 0,
            max_time: int = 5
    ):

        if file is None:
            tmpdir_file_path = os.path.join(os.environ.get('TMPDIR', ''), 'N128_64.nc')
            if os.path.exists(tmpdir_file_path):
                print(f"Using file from local scratch: {tmpdir_file_path}")
                self.file_path = tmpdir_file_path
            else:
                print("Using default file path")
                self.file_path = '/cluster/work/math/camlab-data/data/incompressible/tg/N128_64.nc'

        self.file = netCDF4.Dataset(self.file_path, 'r')

        super().__init__(start=start, training_samples=self.file.variables['u'].shape[0])

        self.input_channel = 3
        self.output_channel = 3

        self.spatial_resolution = (64, 64, 64)
        self.input_shape = (6, 64, 64, 64)
        self.output_shape = (3, 64, 64, 64)

        # these stats can be used to get the mean and std for normalization
        training_stats_path = f"/cluster/work/math/camlab-data/data/diffusion_project/TrainingStats_nothing_DataIC_3D_Time_TG"
        mean_stats_path = os.path.join(training_stats_path, 'mean_99000_0_64_False.npy') # pixel wise mean over every cahnnel
        std_stats_path = os.path.join(training_stats_path, 'std_99000_0_64_False.npy') # pixel wise std over every channel
        mean_data = np.load(mean_stats_path)
        std_data = np.load(std_stats_path)

        mean_vals = mean_data.mean(axis=(0, 1, 2)) # mean over all channels
        std_vals = np.mean(std_data ** 2, (0, 1, 2)) ** 0.5 # std over all channels

        # first 3 channels are for the initial conditions
        self.mean_training_input = mean_vals[:self.input_channel] 
        self.std_training_input = std_vals[:self.input_channel] 
        # last 3 channels are for the output (results)
        self.mean_training_output = mean_vals[self.input_channel:]
        self.std_training_output = std_vals[self.input_channel:]

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
        
        # Concatenate along the last axis to form the output tensor
        output_tensor = np.concatenate(
            (initial_condition, final_condition), axis=-1)  # Shape: (64, 64, 64, 6)
        
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

    def __init__(self,
                 training_samples: int,
                 start: int = 0,
                 micro_perturbation: int = 0,
                 macro_perturbation: int = 0,
                 file_path: str = None,
                 stat_folder: str = None,
                 t_final: int = None) -> None : 

        super().__init__(training_samples=training_samples, start=start)

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
    def __init__(self):
        
        self.input_channel = 2
        self.output_channel = 2

        self.spatial_resolution = (128, 128)
        self.input_shape = (4, 128, 128)
        self.output_shape = (2, 128, 128)
        
        file_path = f'{DIR_PATH_LOADER}/macro_micro_id_2d.nc'
        stat_folder = f'{DIR_PATH_LOADER}/GroundTruthStats_ConditionalDataIC_Vel_nothing_128_10000'

        self.file = netCDF4.Dataset(file_path, mode='r')
        self.start_index = 0    

        samples_shape = self.file.variables['data'].shape
        training_samples = samples_shape[0] * samples_shape[1] # macro * micro perturbations

        super().__init__(training_samples=training_samples,
                         start=self.start_index,
                         micro_perturbation = 1000,
                         macro_perturbation = 10,
                         file_path = file_path,
                         stat_folder = stat_folder)  
         
        self.mean_training_input = np.array([8.0606696e-08, 4.8213877e-11])
        self.std_training_input = np.array([0.19003302, 0.13649726])
        self.mean_training_output = np.array([4.9476512e-09, -1.5097612e-10])
        self.std_training_output = np.array([0.35681796, 0.5053845]) 
    
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

        return {
            # lead time needs to be changed depending on your start index
            'lead_time': torch.tensor(1., dtype=torch.float32),
            'initial_cond': initial_cond,
            'target_cond': target_cond
        }
    

class ConditionalDataIC_3D(ConditionalBase):
    def __init__(self):

        self.input_channel = 3
        self.output_channel = 3

        self.spatial_resolution = (64, 64, 64)
        self.input_shape = (6, 64, 64, 64)
        self.output_shape = (3, 64, 64, 64)

        self.macro_perturbation = 10
        self.micro_perturbation = 1000

        self.time_step = -1

        self.file_path = '/cluster/work/math/camlab-data/data/diffusion_project/macro_micro_id_3d.nc'
        self.data = netCDF4.Dataset(self.file_path, 'r')

        # shape of the dataset: (macro, micro, time, x, y, z, c)
        data_shape = self.data['data'].shape
        training_samples = data_shape[0] * data_shape[1]
        micro_perturbations = data_shape[1]
        macro_perturbations = data_shape[0]

        super().__init__(
            training_samples=training_samples,
            micro_perturbation=micro_perturbations,
            macro_perturbation=macro_perturbations,
            file_path=self.file_path,
        )

        # Set the same mean and std values as for training DataIC_3D_Time
        self.mean_training_input = np.array([1.5445266e-08, 1.2003070e-08, -3.2182508e-09])
        self.mean_training_output = np.array([-8.0223117e-09, -3.3674191e-08, 1.5241447e-08])

        self.std_training_input = np.array([0.20691067, 0.15985465, 0.15808222])
        self.std_training_output = np.array([0.2706984, 0.24893111, 0.24169469])


    def __getitem__(self, index):
        macro_idx = self.get_macro_index(index)
        micro_idx = self.get_micro_index(index)
        datum = self.data.variables['data'][macro_idx, micro_idx, (0, self.time_step)].data

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

        return {
            # lead_time needs to be changed depending on your start index
            'lead_time': torch.tensor(1., dtype=torch.float32),
            'initial_cond': initial_cond,
            'target_cond': target_cond
        }


class ConditionalDataIC_3D_TG(ConditionalBase):
    def __init__(self, t_final: int = 5):

        self.input_channel = 3
        self.output_channel = 3

        self.spatial_resolution = (64, 64, 64)
        self.input_shape = (6, 64, 64, 64)
        self.output_shape = (3, 64, 64, 64)

        self.t_final = t_final

        self.start = 0
        self.start_index = 0
        
        self.file_path = '/cluster/work/math/camlab-data/data/incompressible/tg/micro_ref_sol_N128_64.nc'
        print(f"Start dataset from index {self.start}. Getting data from {self.file_path}")
        self.data = netCDF4.Dataset(self.file_path, 'r')

        u_shape = self.data.variables['u'].shape  # Shape: (6, 64, 64, 64)
        v_shape = self.data.variables['v'].shape
        w_shape = self.data.variables['w'].shape
        assert (u_shape == v_shape == w_shape), "Data needs to align in terms of shape!"
        data_shape = u_shape

        training_samples = data_shape[0] * data_shape[1]
        micro_perturbations = data_shape[1]
        macro_perturbations = data_shape[0]

        super().__init__(
            training_samples=training_samples,
            start=self.start,
            micro_perturbation=micro_perturbations,
            macro_perturbation=macro_perturbations,
            file_path=self.file_path,
            # stat_folder=stat_folder,
            t_final=self.t_final
        )
    
    def __getitem__(self, index):
        macro_idx = self.get_macro_index(index + self.start_index)
        micro_idx = self.get_micro_index(index + self.start_index)

        # Shape: (64, 64, 64)
        u_data = self.data.variables['u'][macro_idx, micro_idx, 0]
        v_data = self.data.variables['v'][macro_idx, micro_idx, 0]
        w_data = self.data.variables['w'][macro_idx, micro_idx, 0]

        # Stack along the new last dimension (axis=-1), Shape: (64, 64, 64, 3)
        data_input = np.stack((u_data, v_data, w_data), axis=-1) 

        # Load the data for the given index, Shape: (64, 64, 64)
        u_data = self.data.variables['u'][macro_idx, micro_idx, self.t_final]
        v_data = self.data.variables['v'][macro_idx, micro_idx, self.t_final]
        w_data = self.data.variables['w'][macro_idx, micro_idx, self.t_final]
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
        
        return {
            'lead_time': torch.tensor(lead_time_normalized, dtype=torch.float32),
            'initial_cond': initial_cond,
            'target_cond': target_cond
        }
    
    
    def set_true_stats_for_macro(self, macro_idx):
        file_name = f"/cluster/work/math/camlab-data/data/diffusion_project/GroundTruthStats_ConditionalDataIC_3D_TG_nothing_64_80000_{self.t_final}/Stats_{macro_idx}.nc"

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