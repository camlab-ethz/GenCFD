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
This module contains all 3D incompressible fluid flow datasets used for training and evaluating diffusion models.

### Overview:
- **Training Datasets:** Standard datasets used to train diffusion models on 3D incompressible flows.  
- **Conditional Datasets:** Datasets prefixed with *Conditional* represent perturbed ensembles, 
        designed for evaluating model generalization and robustness.

### Training Strategy:
All 3D datasets in this module utilize a *lead time* parameter, which conditions the diffusion model on the number 
of timesteps moving forward from the initial condition. An **All-to-All (A2A) training strategy** is applied, 
allowing the model to learn dynamics across all possible temporal pairs.

> ⚠️ *Explicit usage of `lead_time` normalization for specific datasets is defined in* `utils/gencfd_utils.py`.

### Available Datasets:

- **3D Shear Layer:**  
  - `ShearLayer3D`  
  - `ConditionalShearLayer3D`  

- **3D Taylor Green Vortex:**  
  - `TaylorGreen3D`  
  - `ConditionalTaylorGreen3D`  

- **3D Nozzle Flow:**  
  - `Nozzle3D`  
  - `ConditionalNozzle3D` 
"""

import numpy as np
import torch
from dataloader.dataset import TrainingSetBase, ConditionalBase
from typing import Union, Tuple, Any, List, Dict

array = np.ndarray


class LeadTimeNormalizer:
    """
    Handles dataset-specific lead time normalization for 3D diffusion models.

    The model is conditioned on the lead time using an All2All training strategy. 
    Different datasets require different normalization strategies to optimize performance.

    Attributes:
        dataset_name (str): The name of the dataset to determine the normalization strategy.
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def normalize_lead_time(self, t_init: int, t_final: int):
        """
        Applies dataset-specific normalization to the lead time.

        Args:
            t_init (int): The initial timestep.
            t_final (int): The final timestep.

        Returns:
            float: The normalized lead time.
        """

        if self.dataset_name in ['Nozzle3D', 'ConditionalNozzle3D']:
            return self.nozzle3d_normalization(t_init, t_final)

        elif self.dataset_name in ['ShearLayer3D', 'ConditionalShearLayer3D']:
            return self.shearlayer3d_normalization(t_init, t_final)

        elif self.dataset_name in ['TaylorGreen3D, ConditionalTaylorGreen3D']: 
            return self.taylorgreen_normalization(t_init, t_final)

        else:
            return self.default_normalization(t_init, t_final)

    def nozzle3d_normalization(self, t_init: int, t_final: int):
        """Lead time normalized to be in between 0.1 and 1.4"""
        lead_time = t_final - t_init
        return 0.1 * lead_time

    def shearlayer3d_normalization(self, t_init: int, t_final: int):
        """Normalize Values to be in between 0.25 and 1"""
        lead_time = float(t_final - t_init)
        return 0.25 * lead_time

    def taylorgreen_normalization(self, t_init: int, t_final: int): 
        """Normalize lead time to be in between 0.6875 and 2"""
        lead_time = float(t_final - t_init)
        return 0.25 + 0.4375 * (lead_time - 1)

    def default_normalization(self, t_init: int, t_final: int):
        """Default is no normalization but rather computing the lead time"""
        lead_time = t_final - t_init
        return lead_time


class IncompressibleFlows3D(TrainingSetBase):
    """
    Dataset class for 3D incompressible fluid flow simulations used in training diffusion models.

    This class handles the loading, normalization, and preprocessing of 3D incompressible flow datasets
    for model training. It supports flexible data handling, including moving files to local scratch storage,
    custom normalization, and generating all possible (t_initial, t_final) time pairs for All-to-All (A2A) 
    training strategies.

    Attributes:
        min_time (int): The starting timestep for data sampling.
        max_time (int): The final timestep for data sampling.
        time_pairs (List[Tuple[int, int]]): Precomputed (t_initial, t_final) pairs for lead time conditioning.
        total_pairs (int): Total number of (t_initial, t_final) time pairs.
        lead_time_normalizer (LeadTimeNormalizer): Normalizer for lead time values.

    Args:
        file_system (dict): File paths and metadata for dataset storage and retrieval.
        input_channel (int): Number of input channels (variables) for the model.
        output_channel (int): Number of output channels (variables) for the model.
        spatial_resolution (Tuple): Spatial resolution of the dataset (e.g., grid size).
        input_shape (Tuple): Shape of the input data tensors.
        output_shape (Tuple): Shape of the output data tensors.
        variable_names (List[str]): Names of physical variables in the dataset (e.g., velocity, pressure).
        min_time (int): Minimum timestep to include in the dataset.
        max_time (int): Maximum timestep to include in the dataset.
        ndim (int, optional): Number of spatial dimensions (default is 3).
        start (int, optional): Offset to start reading data samples (default is 0).
        training_samples (int, optional): Number of training samples to use. If None, uses all available samples.
        move_to_local_scratch (bool, optional): If True, moves dataset to local scratch storage for faster access.
        retrieve_stats_from_file (bool, optional): If True, loads normalization statistics from a file.
        mean_training_input (array, optional): Mean values for input normalization.
        std_training_input (array, optional): Standard deviation values for input normalization.
        mean_training_output (array, optional): Mean values for output normalization.
        std_training_output (array, optional): Standard deviation values for output normalization.
        get_values (bool, optional): If True, retrieves raw mean and std values from the dataset instead of computing them.

    Raises:
        ValueError: If `min_time` is greater than `max_time`.
        FileNotFoundError: If dataset files specified in `file_system` are not found.

    Example:
        ```python
        dataset = IncompressibleFlows3D(
            file_system = {
                "dataset_name": "TaylorGreen3D", 
                'file_name': 'taylor_green.nc',
                'origin': '/cluster/data/taylor_green/'
                # Additional not relevant file_system settings in case the mean and std were accumulated
                'stats_file': 'GroundTruthStats_ConditionalTaylorGreen',
                'origin_stats': '/cluster/data/taylor_green/'
                # If the the single mean and std values in vector form are available
                'mean_file': 'mean.npy',
                'std_file': 'std.npy',
            }
        )
        sample = dataset.__getitem__(0)
        print(sample)
        ```
    """

    def __init__(
            self,
            file_system: dict,
            input_channel: int,
            output_channel: int,
            spatial_resolution: Tuple,
            input_shape: Tuple,
            output_shape: Tuple,
            variable_names: List[str],
            min_time: int,
            max_time: int,
            ndim: int = 3,
            start: int = 0,
            training_samples: int = None,
            move_to_local_scratch: bool = False,
            retrieve_stats_from_file: bool = False,
            mean_training_input: array = None,
            std_training_input: array = None,
            mean_training_output: array = None,
            std_training_output: array = None,
            get_values: bool = False
        ) -> None: 

        super().__init__(
            file_system=file_system,
            ndim=ndim, # Always 3D dataset
            input_channel=input_channel,
            output_channel=output_channel,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names,
            start=start,
            training_samples=training_samples,
            move_to_local_scratch=move_to_local_scratch,
            retrieve_stats_from_file=retrieve_stats_from_file,
            mean_training_input=mean_training_input,
            std_training_input=std_training_input,
            mean_training_output=mean_training_output,
            std_training_output=std_training_output,
            get_values=get_values
        )

        self.min_time = min_time
        self.max_time = max_time

        # Precompute all possible (t_initial, t_final) pairs within the specified range.
        self.time_pairs = [(i, j) for i in range(self.min_time, self.max_time) for j in range(i + 1, self.max_time + 1)]
        self.total_pairs = len(self.time_pairs)

        # get the correct normalization method for the lead time
        self.lead_time_normalizer = LeadTimeNormalizer(dataset_name=self.file_system['dataset_name'])
    

    def normalize_lead_time(self, t_init: int, t_final: int) -> Union[int, float]:
        """Uses the correct normalization scheme for both the Conditional and Training Dataset"""
        return self.lead_time_normalizer.normalize_lead_time(t_init, t_final)


    def __len__(self):
        # Return the total number of data points times the number of pairs.
        return self.training_samples * self.total_pairs
    

    def __getitem__(self, index):
        # Determine the data point and the (t_initial, t_final) pair
        data_index = index // self.total_pairs
        pair_index = index % self.total_pairs
        t_init, t_final = self.time_pairs[pair_index]

        # List with all variables relevant for the given dataset will be stacked later
        data_list = [self.file.variables[var][data_index] for var in self.variable_names]
        if self.file_system['dataset_name'] == 'Nozzle3D':
            # Add an additional conditioning tensor is required for the Nozzle dataset
            vel_inject = float(self.file.variables['injection_velocity'][data_index])*np.ones((192, 64, 64))
            data_list = [np.broadcast_to(vel_inject, (14, 192, 64, 64))] + data_list
        
        combined_data = np.stack(data_list, axis=-1)

        # Extract initial and final conditions
        initial_condition = self.normalize_input(
            combined_data[t_init]) 

        if self.file_system['dataset_name'] == 'Nozzle3D':
            combined_data = combined_data[..., 1:] # get rid of the conditioning
        
        final_condition = self.normalize_output(
            combined_data[t_final])
        
        lead_time_normalized = self.normalize_lead_time(t_init=t_init, t_final=t_final)

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
        

class ShearLayer3D(IncompressibleFlows3D):
    def __init__(
            self,
            metadata: Dict[str, Any],
            start=0,
            move_to_local_scratch: bool = True, 
            retrieve_stats_from_file: bool = False,
            input_channel: int = 3,
            output_channel: int = 3,
            spatial_resolution: Tuple[int, ...] = (64, 64, 64),
            input_shape: Tuple[int, ...] = (6, 64, 64, 64),
            output_shape: Tuple[int, ...] = (3, 64, 64, 64),
            variable_names: List[str] = ['u', 'v', 'w'],
            min_time: int = 0,
            max_time: int = 4,
            mean_training_input: array = np.array([1.5445266e-08, 1.2003070e-08, -3.2182508e-09]),
            mean_training_output: array = np.array([-8.0223117e-09, -3.3674191e-08, 1.5241447e-08]),
            std_training_input: array = np.array([0.20691067, 0.15985465, 0.15808222]),
            std_training_output: array = np.array([0.2706984, 0.24893111, 0.24169469])
        ):

        super().__init__( 
            file_system=metadata,
            input_channel=input_channel,
            output_channel=output_channel,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names,
            min_time=min_time,
            max_time=max_time,
            start=start,
            move_to_local_scratch=move_to_local_scratch,
            retrieve_stats_from_file=retrieve_stats_from_file,
            mean_training_input=mean_training_input,
            std_training_input=std_training_input,
            mean_training_output=mean_training_output,
            std_training_output=std_training_output
        )

    def normalize_lead_time(self, t_init: int, t_final: int) -> Union[int, float]:
        """Normalize Values to be in between 0 and 1"""
        lead_time = float(t_final - t_init)
        return 0.25 * lead_time


class TaylorGreen3D(IncompressibleFlows3D):
    
    def __init__(
            self,
            metadata: Dict[str, Any],
            start: int = 0,
            min_time: int = 0,
            max_time: int = 5,
            move_to_local_scratch: bool = True, 
            retrieve_stats_from_file: bool = True,
            input_channel: int = 3,
            output_channel: int = 3,
            spatial_resolution: Tuple[int, ...] = (64, 64, 64),
            input_shape: Tuple[int, ...] = (6, 64, 64, 64),
            output_shape: Tuple[int, ...] = (3, 64, 64, 64),
            variable_names: List[str] = ['u', 'v', 'w']
    ):

        super().__init__(
            file_system=metadata,
            input_channel=input_channel,
            output_channel=output_channel,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names,
            min_time=min_time,
            max_time=max_time,
            start=start,
            move_to_local_scratch=move_to_local_scratch,
            retrieve_stats_from_file=retrieve_stats_from_file
        )


class Nozzle3D(IncompressibleFlows3D):

    def __init__(
        self,
        metadata: Dict[str, Any],
        start: int = 0,
        file: str = None,
        min_time: int = 0,
        max_time: int = 14,
        move_to_local_scratch: bool = False,
        retrieve_stats_from_file: bool = False,
        input_channel: int = 4,
        output_channel: int = 3,
        spatial_resolution: Tuple[int, ...] = (64, 64, 192),
        input_shape: Tuple[int, ...] = (4, 64, 64, 192),
        output_shape: Tuple[int, ...] = (3, 64, 64, 192),
        variable_names: List[str] = ['u', 'v', 'w'],
        mean_training_input: array = np.array([0.0, 0.00858, 0.0, 0.0]),
        std_training_input: array = np.array([1.0, 0.0727, 0.0266, 0.0252]),
        mean_training_output: array = np.array([0.00858, 0.0, 0.0]),
        std_training_output: array = np.array([0.0727, 0.0266, 0.0252])
    ):

        super().__init__(
            file_system=metadata,
            input_channel=input_channel,
            output_channel=output_channel,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names,
            min_time=min_time,
            max_time=max_time,
            start=start,
            move_to_local_scratch=move_to_local_scratch,
            retrieve_stats_from_file=retrieve_stats_from_file,
            mean_training_input=mean_training_input,
            std_training_input=std_training_input,
            mean_training_output=mean_training_output,
            std_training_output=std_training_output
        )

        # Overwrite pairs since here every second step should be taken
        self.time_pairs = [(2*i, 2*j) for i in range(0, self.max_time//2 - 1) for j in range(i + 1, self.max_time//2)]
        self.total_pairs = len(self.time_pairs)


class ConditionalIncompressibleFlows3D(ConditionalBase):
    """
    Conditional dataset class for 3D incompressible fluid flow simulations with perturbations.

    This class extends the functionality of `IncompressibleFlows3D` by introducing micro and macro 
    perturbations for evaluating the robustness of diffusion models. It is designed for testing how 
    the model handles perturbed initial conditions over a specified lead time.

    For general dataset handling and attributes, refer to `IncompressibleFlows3D`.

    Attributes:
        t_start (int): The initial timestep for sampling.
        t_final (int): The final timestep for sampling.
        lead_time_normalizer (LeadTimeNormalizer): Normalizer for the lead time between `t_start` and `t_final`.

    Args:
        micro_perturbations (int): Number of micro-scale perturbations applied to the dataset for fine-grained variability.
        macro_perturbations (int): Number of macro-scale perturbations applied to the dataset for large-scale variability.
        t_start (int): The starting timestep for conditional sampling.
        t_final (int): The ending timestep for conditional sampling.
        training_samples (int, optional): Number of training samples to use. If None, uses all available samples.

    Example:
        See IncompressibleFlows3D
    """

    def __init__(
            self,
            file_system: dict,
            input_channel: int,
            output_channel: int,
            spatial_resolution: Tuple,
            input_shape: Tuple,
            output_shape: Tuple,
            variable_names: List[str],
            micro_perturbations: int,
            macro_perturbations: int,
            t_start: int,
            t_final: int,
            start: int = 0,
            ndim: int = 3,
            training_samples: int = None,
            move_to_local_scratch: bool = False,
        ) -> None: 

        super().__init__(
            file_system=file_system,
            ndim=ndim, # Always 3D dataset
            input_channel=input_channel,
            output_channel=output_channel,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names,
            start=start,
            training_samples=training_samples,
            move_to_local_scratch=move_to_local_scratch,
            micro_perturbations=micro_perturbations,
            macro_perturbations=macro_perturbations
        )

        self.t_start = t_start
        self.t_final = t_final

        # get the correct normalization method for the lead time
        self.lead_time_normalizer = LeadTimeNormalizer(dataset_name=self.file_system['dataset_name'])

    def normalize_lead_time(self, t_init: int, t_final: int) -> Union[int, float]:
        """Uses the correct normalization scheme for both the Conditional and Training Dataset"""
        return self.lead_time_normalizer.normalize_lead_time(t_init, t_final)
    

    def __getitem__(self, index):
        
        macro_idx = self.get_macro_index(index + self.start)
        micro_idx = self.get_micro_index(index + self.start)

        # Preload selector since some datasets have only 1 macro perturbation included
        idx_selector = (macro_idx, micro_idx) if self.macro_perturbations > 1 else (micro_idx,)

        # Stack along the new last dimension (axis=-1) and dynamically load the data
        data_initial = [self.file.variables[var][*idx_selector, self.t_start] for var in self.variable_names]
        data_target = [self.file.variables[var][*idx_selector, self.t_final] for var in self.variable_names]

        if self.file_system['dataset_name'] == 'ConditionalNozzle3D':
            # Additional Conditioning for the Nozzle Dataset
            vel_inject = float(self.file.variables['injection_velocity'][micro_idx])*np.ones((192, 64, 64))
            data_initial.insert(0, vel_inject)

        data_input = np.stack(data_initial, axis=-1)
        data_output = np.stack(data_target, axis=-1)


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

        lead_time_normalized = self.normalize_lead_time(t_init=self.t_start, t_final=self.t_final)
        
        # Store indices for the CDF computation
        return {
            'lead_time': torch.tensor(lead_time_normalized, dtype=torch.float32),
            'initial_cond': initial_cond,
            'target_cond': target_cond
        }
        

class ConditionalShearLayer3D(ConditionalIncompressibleFlows3D):
    def __init__(
            self, 
            metadata: Dict[str, Any],
            move_to_local_scratch: bool = True, 
            input_channel: int = 3,
            output_channel: int = 3,
            spatial_resolution: Tuple[int, ...] = (64, 64, 64),
            input_shape: Tuple[int, ...] = (6, 64, 64, 64),
            output_shape: Tuple[int, ...] = (3, 64, 64, 64),
            variable_names: List[str] = ['u', 'v', 'w'],
            macro_perturbations: int = 10,
            micro_perturbations: int = 1000,
            t_start: int = 0,
            t_final: int = 4,
            start: int = 0,
        ):

        super().__init__(
            file_system=metadata,
            training_samples=micro_perturbations * macro_perturbations,
            input_channel=input_channel,
            output_channel=output_channel,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names,
            start=start,
            move_to_local_scratch=move_to_local_scratch,
            micro_perturbations=micro_perturbations,
            macro_perturbations=macro_perturbations,
            t_start=t_start,
            t_final=t_final
        )


class ConditionalTaylorGreen3D(ConditionalIncompressibleFlows3D):
    def __init__(
            self, 
            metadata: Dict[str, Any],
            move_to_local_scratch: bool = False,
            input_channel: int = 3,
            output_channel: int = 3,
            spatial_resolution: Tuple[int, ...] = (64, 64, 64),
            input_shape: Tuple[int, ...] = (6, 64, 64, 64),
            output_shape: Tuple[int, ...] = (3, 64, 64, 64),
            variable_names: List[str] = ['u', 'v', 'w'],
            macro_perturbations: int = 10,
            micro_perturbations: int = 1000,
            t_start: int = 0,
            t_final: int = 5,
            start: int = 0
        ):

        super().__init__(
            training_samples=micro_perturbations * macro_perturbations,
            file_system=metadata,
            move_to_local_scratch=move_to_local_scratch,
            input_channel=input_channel,
            output_channel=output_channel,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names,
            micro_perturbations=micro_perturbations,
            macro_perturbations=macro_perturbations,
            t_start=t_start,
            t_final=t_final,
            start=start
        )


class ConditionalNozzle3D(ConditionalIncompressibleFlows3D):
    def __init__(
            self, 
            metadata: Dict[str, Any], 
            move_to_local_scratch: bool = False,
            input_channel: int = 4,
            output_channel: int = 3,
            spatial_resolution: Tuple[int, ...] = (64, 64, 192),
            input_shape: Tuple[int, ...] = (4, 64, 64, 192),
            output_shape: Tuple[int, ...] = (3, 64, 64, 192),
            variable_names: List[str] = ['u', 'v', 'w'],
            macro_perturbations: int = 1,
            micro_perturbations: int = 4000,
            t_start: int = 0,
            t_final: int = 10,
            start: int = 0
        ):

        super().__init__(
            training_samples=micro_perturbations * macro_perturbations,
            file_system=metadata,
            move_to_local_scratch=move_to_local_scratch,
            input_channel=input_channel,
            output_channel=output_channel,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names,
            micro_perturbations=micro_perturbations,
            macro_perturbations=macro_perturbations,
            t_start=t_start,
            t_final=t_final,
            start=start
        )
    
    def get_mask(self):
        """Used to mask the output"""
        mask = self.file.variables['mask'][:]
        return (
            torch.from_numpy(mask)
            .type(torch.float32)
            .permute(2, 1, 0)
        )