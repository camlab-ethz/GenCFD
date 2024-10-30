import os
import netCDF4
import numpy as np
import torch
from typing import Union

DIR_PATH_LOADER = '/cluster/work/math/camlab-data/data/diffusion_project'

# data = netCDF4.Dataset('/cluster/home/yramic/swirl-dynamics-torch/datasets/Stats_0.nc', 'r')

array = np.ndarray
Tensor = torch.Tensor


class TrainingSetBase:
    def __init__(self,
                 training_samples: int,
                 start: int = 0,
                 device: torch.device = None) -> None:
        
        self.start = start
        self.training_samples = training_samples
        self.device = device
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

#--------------------------------------

class DataIC_Vel(TrainingSetBase):
    def __init__(self, 
                 start: int = 0,
                 file: str = None,
                 device: torch.device = None):
        
        self.class_name = self.__class__.__name__
        self.input_channel = 2
        self.output_channel = 2
        self.spatial_dim = 2
        
        if file is None:
            self.file = netCDF4.Dataset(f'{DIR_PATH_LOADER}/ddsl_fast_nothing_128_tr2.nc', mode='r')
        else:
            self.file = file

        super().__init__(start = start, device=device, training_samples=self.file['data'].shape[0])
        
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

        model_input = torch.cat(
            [torch.as_tensor(data_input, dtype=torch.float32, device=self.device), 
             torch.as_tensor(data_output, dtype=torch.float32, device=self.device)], 
            dim=-1
        )

        model_input = model_input.permute(2, 1, 0)

        return model_input
    

class DataIC_3D_Time(TrainingSetBase):
    def __init__(
            self,
            start=0,
            device=None,
            file = None
        ):

        self.input_channel = 3
        self.output_channel = 3

        if file is None:
            if start == 0:
                self.file_path = '/cluster/work/math/camlab-data/data/diffusion_project/shear_layer_3D_64_all_time_2.nc'
            else:
                self.file_path = '/cluster/work/math/camlab-data/data/diffusion_project/shear_layer_3D_64_smaller.nc'

        self.file = netCDF4.Dataset(self.file_path, 'r')

        super().__init__(start=start, device=device, training_samples=self.file.variables['data'].shape[0])

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

        inputs = np.concatenate((data_input, data_output), -1)

        return (
            torch.tensor(lead_time, dtype=torch.float32, device=self.device), 
            torch.tensor(inputs, dtype=torch.float32, device=self.device).permute(3, 2, 1, 0)
        )

    def collate_tf(self, time, data):
        return {"lead_time": time, "data": data}

    def get_proc_data(self, data):
        return data["data"]


#----------------------------------

class ConditionalBase(TrainingSetBase):

    def __init__(self,
                 training_samples: int,
                 start: int = 0,
                 device: torch.device = None,
                 micro_perturbation: int = 0,
                 macro_perturbation: int = 0,
                 file_path: str = None,
                 stat_folder: str = None,
                 t_final: int = None) -> None : 

        super().__init__(training_samples=training_samples, start=start, device=device)

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
    def __init__(self, device: torch.device = None):
        
        self.input_channel = 2
        self.output_channel = 2
        self.spatial_dim = 2
        
        file_path = f'{DIR_PATH_LOADER}/macro_micro_id_2d.nc'
        stat_folder = f'{DIR_PATH_LOADER}/GroundTruthStats_ConditionalDataIC_Vel_nothing_128_10000'

        self.file = netCDF4.Dataset(file_path, mode='r')
        self.start_index = 0    

        samples_shape = self.file.variables['data'].shape
        training_samples = samples_shape[0] * samples_shape[1] # macro * micro perturbations

        super().__init__(training_samples=training_samples,
                         start=self.start_index,
                         device=device,
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

        model_input = torch.cat(
            [torch.as_tensor(data_input, dtype=torch.float32), 
             torch.as_tensor(data_output, dtype=torch.float32)], 
            dim=-1
        )
        model_input = model_input.permute(2, 1, 0)

        return model_input
    

class ConditionalDataIC_3D(ConditionalBase):
    def __init__(self, device: torch.device = None):

        self.input_channel = 3
        self.output_channel = 3

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
            device=device,
            micro_perturbation=micro_perturbations,
            macro_perturbation=macro_perturbations,
            file_path=self.file_path,
            # stat_folder=stat_folder,
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

        model_input = torch.cat(
            [torch.as_tensor(data_input, dtype=torch.float32), 
             torch.as_tensor(data_output, dtype=torch.float32)], 
            dim=-1
        )
        model_input = model_input.permute(3, 2, 1, 0)

        return model_input


class DataIC_Vel_Test(TrainingSetBase):
    def __init__(self, 
                 training_samples = 100,
                 start = 0,
                 file = None):
        
        super().__init__(training_samples, start = start)
        
        self.class_name = self.__class__.__name__
        self.input_channel = 2
        self.output_channel = 2

        self.file = {'data': torch.randn((1000, 2, 32, 32, 3))}
        

    def __getitem__(self, index):
        """Load data from disk on the fly given an index"""

        index += self.start       
        data = self.file['data'][index].data

        data_input = data[0, ..., :self.input_channel]
        data_output = data[1, ..., :self.output_channel]

        inputs = torch.cat((data_input, data_output), -1)
        
        return inputs.permute(2, 1, 0)
    
    def __len__(self):
        return self.file['data'].shape[0]


class MNIST_Test(TrainingSetBase):
    def __init__(self, 
                 training_samples = 100,
                 start = 0,
                 file = None):
        
        super().__init__(training_samples, start = start)
        
        self.class_name = self.__class__.__name__
        self.input_channel = 1
        self.output_channel = 1

        self.file = {'data': torch.randn((1000, 2, 28, 28, 2))}
        

    def __getitem__(self, index):
        """Load data from disk on the fly given an index"""

        index += self.start       
        data = self.file['data'][index].data

        data_input = data[0, ..., :self.input_channel]
        data_output = data[1, ..., :self.output_channel]

        inputs = torch.cat((data_input, data_output), -1)
        
        return inputs.permute(2, 1, 0)
    
    def __len__(self):
        return self.file['data'].shape[0]