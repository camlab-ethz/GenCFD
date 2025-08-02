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
This module contains all 2D incompressible fluid flow datasets used for training and evaluating diffusion models.

### Overview:
- **Training Datasets:** Standard datasets used to train diffusion models on 2D incompressible flows.  
- **Conditional Datasets:** Datasets prefixed with *Conditional* represent perturbed ensembles, 
        designed for evaluating model generalization and robustness.

### Training Strategy:
All 2D datasets in contrast to the 3D datasets don't utilize a *lead time* parameter, which conditions the diffusion model on the number 
of timesteps moving forward from the initial condition.

> ⚠️ *Explicit usage of `lead_time` normalization for specific datasets is defined in* `utils/gencfd_utils.py`.

### Available Datasets:

- **2D Shear Layer:**  
  - `ShearLayer2D`  
  - `ConditionalShearLayer2D`  

- **2D Cloud Shock:**  
  - `CloudShock2D`  
  - `ConditionalCloudShock2D`  

- **2D RichtmyerMeshkov2D:**  
  - `RichtmyerMeshkov2D`  
  - `RichtmyerMeshkov2D` 
  - There is no seperate Conditional Dataset
"""
import numpy as np
import torch
from GenCFD.dataloader.dataset import TrainingSetBase, ConditionalBase
from typing import Tuple, Any, List, Dict

array = np.ndarray


class IncompressibleFlows2D(TrainingSetBase):
    """
    Dataset class for 2D incompressible fluid flow simulations used in training diffusion models.

    This class handles the loading, normalization, and preprocessing of 2D incompressible flow datasets
    for model training. It supports flexible data handling, including moving files to local scratch storage.

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
        FileNotFoundError: If dataset files specified in `file_system` are not found.

    Example:
        ```python
        dataset = IncompressibleFlows2D(
            file_system = {
                "dataset_name": "ShearLayer2D",
                'file_name': 'shear_layer_2d.nc',
                'origin': '/cluster/data/shear_layer/'
                # Additional not relevant file_system settings in case the mean and std were accumulated
                'stats_file': 'GroundTruthStats_ShearLayer2D',
                'origin_stats': '/cluster/data/shear_layer/'
                # If the the single mean and std values in vector form are available
                'mean_file': 'mean.npy',
                'std_file': 'std.npy',
            }
        )
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
        ndim: int = 2,
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

        super().__init__(
            file_system=file_system,
            ndim=ndim,  # Always 2D dataset
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
            get_values=get_values,
        )


    def __getitem__(self, index):

        index += self.start
        # List with all variables relevant for the given dataset will be stacked later

        if self.file_system["dataset_name"] != 'RichtmyerMeshkov2D':
            combined_data = self.file["data"][index].data
        else:
            data_list = [
                self.file.variables[var][index] for var in self.variable_names
            ]
            combined_data = np.stack(data_list, axis=-1)

        # Extract initial and final conditions
        initial_condition = self.normalize_input(
            combined_data[0, ..., :self.input_channel]
        ) 
        target_condition = self.normalize_output(
            combined_data[1, ..., :self.output_channel]
        ) 

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
            "initial_cond": initial_cond,
            "target_cond": target_cond,
        }


class ShearLayer2D(IncompressibleFlows2D):
    def __init__(
        self,
        metadata: Dict[str, Any],
        start=0,
        move_to_local_scratch: bool = False,
        retrieve_stats_from_file: bool = False,
        input_channel: int = 2,
        output_channel: int = 2,
        spatial_resolution: Tuple[int, ...] = (128, 128),
        input_shape: Tuple[int, ...] = (4, 128, 128),
        output_shape: Tuple[int, ...] = (2, 128, 128),
        variable_names: List[str] = ["data"],
        mean_training_input: array = np.array([8.0606696e-08, 4.8213877e-11]),
        mean_training_output: array = np.array([4.9476512e-09, -1.5097612e-10]),
        std_training_input: array = np.array([0.19003302, 0.13649726]),
        std_training_output: array = np.array([0.35681796, 0.5053845]),
    ):

        super().__init__(
            file_system=metadata,
            input_channel=input_channel,
            output_channel=output_channel,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names,
            start=start,
            move_to_local_scratch=move_to_local_scratch,
            retrieve_stats_from_file=retrieve_stats_from_file,
            mean_training_input=mean_training_input,
            std_training_input=std_training_input,
            mean_training_output=mean_training_output,
            std_training_output=std_training_output,
        )


class CloudShock2D(IncompressibleFlows2D):
    def __init__(
        self,
        metadata: Dict[str, Any],
        start=0,
        move_to_local_scratch: bool = False,
        retrieve_stats_from_file: bool = False,
        input_channel: int = 4,
        output_channel: int = 4,
        spatial_resolution: Tuple[int, ...] = (128, 128),
        input_shape: Tuple[int, ...] = (8, 128, 128),
        output_shape: Tuple[int, ...] = (4, 128, 128),
        variable_names: List["str"] = ["data"],
        mean_training_input: array = np.array([1.6308324e+00,  2.2193029e+00, -7.3468456e-16,  3.6186913e+01]),
        mean_training_output: array = np.array([4.2250075e+00,  4.1400085e+01, -3.4726902e-03,  5.9366187e+02]),
        std_training_input: array = np.array([5.3733552e-01, 3.6503599e+00, 3.9603324e-14, 5.5409000e+01]),
        std_training_output: array = np.array([1.5385458, 10.00513  , 3.885837 , 63.43266])
    ):

        self.class_name = self.__class__.__name__

        super().__init__(
            file_system=metadata,
            input_channel=input_channel,
            output_channel=output_channel,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names,
            start=start,
            move_to_local_scratch=move_to_local_scratch,
            retrieve_stats_from_file=retrieve_stats_from_file,
            mean_training_input=mean_training_input,
            std_training_input=std_training_input,
            mean_training_output=mean_training_output,
            std_training_output=std_training_output,
        )


class RichtmyerMeshkov2D(IncompressibleFlows2D):
    def __init__(
        self,
        metadata: Dict[str, Any],
        start=0,
        file=None,
        move_to_local_scratch: bool = False,
        retrieve_stats_from_file: bool = False,
        input_channel: int = 4,
        output_channel: int = 4,
        spatial_resolution: Tuple[int, ...] = (128, 128),
        input_shape: Tuple[int, ...] = (8, 128, 128),
        output_shape: Tuple[int, ...] = (4, 128, 128),
        variable_names: List["str"] = ["rho", "E", "mx", "my"],
        mean_training_input: array = np.array([1.19643337, 3.99017382, 0., 0.]),
        mean_training_output: array = np.array([ 1.19643338e+00,  3.99017382e+00, -1.34351425e-09,  3.37138772e-11]),
        std_training_input: array = np.array([0.39482422, 8.16406155, 0., 0.]),
        std_training_output: array = np.array([0.56974495, 1.66372766, 0.25579005, 0.25579102])
    ):

        self.class_name = self.__class__.__name__

        super().__init__(
            file_system=metadata,
            input_channel=input_channel,
            output_channel=output_channel,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names,
            start=start,
            move_to_local_scratch=move_to_local_scratch,
            retrieve_stats_from_file=retrieve_stats_from_file,
            get_values=True,
            mean_training_input=mean_training_input,
            std_training_input=std_training_input,
            mean_training_output=mean_training_output,
            std_training_output=std_training_output,
        )


class ConditionalShearLayer2D(ConditionalBase):
    def __init__(
        self,
        metadata: Dict[str, Any],
        move_to_local_scratch: bool = False,
        input_channel: int = 2,
        output_channel: int = 2,
        spatial_resolution: Tuple[int, ...] = (128, 128),
        input_shape: Tuple[int, ...] = (4, 128, 128),
        output_shape: Tuple[int, ...] = (2, 128, 128),
        variable_names: List["str"] = ["data"],
        macro_perturbations: int = 10,
        micro_perturbations: int = 1000,
        start: int = 0
    ):

        super().__init__(
            file_system=metadata,
            training_samples=micro_perturbations * macro_perturbations,
            ndim=2,
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
        )

    def __getitem__(self, index):
        macro_idx = self.get_macro_index(index + self.start)
        micro_idx = self.get_micro_index(index + self.start)
        datum = self.file.variables["data"][macro_idx, micro_idx].data

        data_input = datum[0, ..., : self.input_channel]
        data_output = datum[-1, ..., : self.output_channel]

        initial_cond = torch.from_numpy(data_input).type(torch.float32).permute(2, 1, 0)
        target_cond = torch.from_numpy(data_output).type(torch.float32).permute(2, 1, 0)

        return {"initial_cond": initial_cond, "target_cond": target_cond}


class ConditionalCloudShock2D(ConditionalBase):
    def __init__(
        self,
        metadata: Dict[str, Any],
        move_to_local_scratch: bool = False,
        input_channel: int = 4,
        output_channel: int = 4,
        spatial_resolution: Tuple[int, ...] = (128, 128),
        input_shape: Tuple[int, ...] = (8, 128, 128),
        output_shape: Tuple[int, ...] = (4, 128, 128),
        variable_names: List["str"] = ["data"],
        start: int = 0
    ):

        super().__init__(
            training_samples=1000 * 10,
            file_system=metadata,
            input_channel=input_channel,
            output_channel=output_channel,
            start=start,
            micro_perturbations=1000,
            macro_perturbations=10,
            ndim=2,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names
        )

    def __getitem__(self, index):
        index = self.check_indices(index=index)
        macro_idx = self.get_macro_index(index + self.start)
        micro_idx = self.get_micro_index(index + self.start)
        datum = self.file.variables["data"][macro_idx, micro_idx].data

        data_input = datum[0, ..., : self.input_channel]
        data_output = datum[-1, ..., : self.output_channel]

        initial_cond = torch.from_numpy(data_input).type(torch.float32).permute(2, 1, 0)
        target_cond = torch.from_numpy(data_output).type(torch.float32).permute(2, 1, 0)

        return {"initial_cond": initial_cond, "target_cond": target_cond}
    
    def check_indices(self, index: int):
        problematic_indices = [416, 470, 795, 880]
        if index in problematic_indices:
            return index + 1
        else:
            return index
