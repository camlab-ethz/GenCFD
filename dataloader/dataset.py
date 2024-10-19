import os

# import h5py
import netCDF4
import numpy as np
import torch
from typing import Tuple
# import skimage
# from tqdm import tqdm
import time

from utils.dataloader_utils import (
    StatsRecorder, 
    timeit,
    downsample,
    upsample,
    translate_horizontally_periodic_unbatched
)
from dataloader.dataloader import DummyDataloader

Tensor = torch.Tensor

# **********************
# Base Dataset Class
# **********************

class TrainingSetBase:
    def __init__(self,
                 training_samples,
                 start=0):
        
        self.start = start
        self.training_samples = training_samples
        self.rand_gen = np.random.RandomState(seed = 4)
    
        self.mean_training_input = None
        self.std_training_input = None
        self.mean_training_output = None
        self.std_training_output = None
        
        self.input_channel = 1  # Placeholder redefine it in the child class
        self.output_channel = 1  # Placeholder redefine it in the child class

    def __len__(self):
        return self.training_samples

    def normalize_input(self, u_, ):

        if self.mean_training_input is not None:
            return (u_ - self.mean_training_input) / (self.std_training_input + 1e-12)
        else:
            return u_

    def denormalize_input(self, u_, ):
        
        if self.mean_training_input is not None:
            return u_ * (self.std_training_input + 1e-12) + self.mean_training_input
        else:
            return u_

    def normalize_output(self, u_, ):
        if self.mean_training_output is not None:
            return (u_ - self.mean_training_output) / (self.std_training_output + 1e-12)
        else:
            return u_

    def denormalize_output(self, u_, ):
        if self.mean_training_output is not None:
            return u_ * (self.std_training_output + 1e-12) + self.mean_training_output
        else:
            return u_

    def __getitem__(self, item):
        raise NotImplementedError()

    def get_proc_data(self, data):
        return data

    def collate_tf(self, data):
        return data



# **********************
# ALL AVAILABLE DATASETS
# **********************
class DataIC_Vel(TrainingSetBase):
    def __init__(self, 
                 training_samples = 100,
                 start = 0,
                 file = None):
        
        super().__init__(training_samples, start = start)
        
        self.class_name = self.__class__.__name__
        self.input_channel = 2
        self.output_channel = 2

        if file is None:
            self.file = netCDF4.Dataset(f'/cluster/work/math/camlab-data/data/diffusion_project/ddsl_fast_nothing_128_tr2.nc', mode='r')
        else:
            self.file = file

        self.mean_training_input = np.array([8.0606696e-08, 4.8213877e-11])
        self.std_training_input = np.array([0.19003302, 0.13649726])
        self.mean_training_output = np.array([4.9476512e-09, -1.5097612e-10])
        self.std_training_output = np.array([0.35681796, 0.5053845])

    def get_tiny_dataset(self, nsamples: int = None, data_type: str = 'train') -> Tensor:
        
        if nsamples is not None:
            data = self.file["data"][0:nsamples, ...]
        else:
            data = self.file["data"]

        if data_type == 'train':
            data = data[:nsamples//2, ...]

        elif data_type == 'eval':
            data = data[nsamples//2:nsamples, ...]
        else:
            raise ValueError("The dataset type can only be 'train' or 'eval'!")

        data_inp = data[:, 0, ..., :self.input_channel]
        data_out = data[:, 1, ..., :self.output_channel]

        data_inp = self.normalize_input(data_inp)
        data_out = self.normalize_output(data_out)

        model_input = torch.cat(
            [torch.as_tensor(data_inp, dtype=torch.float32), 
             torch.as_tensor(data_out, dtype=torch.float32)], 
            dim=-1
        )
        model_input = model_input.permute(0, 3, 2, 1)
        return model_input
        

    def __getitem__(self, index):
        """Load data from disk on the fly given an index"""
        index += self.start       
        data = self.file['data'][index].data

        data_input = data[0, ..., :self.input_channel]
        data_output = data[1, ..., :self.output_channel]

        data_input = self.normalize_input(data_input)
        data_output = self.normalize_output(data_output)

        inputs = np.concatenate((data_input, data_output), -1)
        
        return torch.tensor(inputs, dtype=torch.float32).permute(2, 1, 0)
    
    def __len__(self):
        return self.file['data'].shape[0]


# data = DataIC_Vel(100)
# dataset = data.get_dataset(200)

# import os

# import h5py
# import netCDF4
# import numpy as np
# import skimage
# from tqdm import tqdm

# from decorators import timeit
# from utilis_stats import StatsRecorderNew as StatsRecorder
# from utils.dataloader_utils import downsample, translate_horizontally_periodic_unbatched, upsample
# from dataloader.dataloader import DummyDataset


# ########################################################################################################################################
# #### Base Class ####
# ########################################################################################################################################
# class TrainingSetBase:
#     def __init__(self,
#                  size,
#                  n_spatial_dim,
#                  which,
#                  training_samples,
#                  learn_residual,
#                  training_variable=None,
#                  mean_training_=None,
#                  std_training_=None,
#                  start=0,
#                  aug_data=0,
#                  mc_samples=None,
#                  workdir=None,
#                  plot=False,
#                  p_dropout=0,
#                  scalar_norm=True):
#         self.size = size
#         self.n_spatial_dim = n_spatial_dim
#         self.which = which
#         self.start = start
#         self.training_samples = training_samples
#         self.aug_data = aug_data
#         self.tot_samples = training_samples + aug_data if mc_samples is None else mc_samples
#         self.learn_residual = learn_residual
#         self.low_sizes = []
#         self.training_variable = training_variable
#         self.rand_gen = np.random.RandomState(seed=42)
#         self.mean_training_ = mean_training_
#         self.std_training_ = std_training_
#         self.mean_training_input = None
#         self.std_training_input = None
#         self.mean_training_output = None
#         self.std_training_output = None
#         self.training_stats_path = workdir
#         self.plot = plot
#         self.input_channel = 1  # Placeholder redefine it in the child class
#         self.output_channel = 1  # Placeholder redefine it in the child class
#         self.stat_size = None
#         self.p_dropout = 0
#         self.use_scalar_norm = scalar_norm
#         print(f"Training Samples: {training_samples}. Starting from {start} with additional {aug_data} translational data")
#         print(f"Total Samples: {self.tot_samples}")
#         print(f"MC Samples: {mc_samples}")

#     def __len__(self):
#         return self.tot_samples

#     def normalize_input(self, u_, ):
#         if self.mean_training_input is not None:
#             # return (u_ - self.mean_training_input) / (self.std_training_input + 1e-16)
#             return (u_ - self.mean_training_input) / (self.std_training_input + 1e-16)
#         else:
#             return u_

#     def normalize_output(self, u_, ):
#         if self.mean_training_output is not None:
#             # return (u_ - self.mean_training_output) / (self.std_training_output + 1e-16)
#             return (u_ - self.mean_training_output) / (self.std_training_output + 1e-16)
#         else:
#             return u_

#     def __getitem__(self, item):
#         raise NotImplementedError()

#     def get_index_shift(self, index):
#         if index < self.training_samples or index > self.training_samples + self.aug_data:
#             return index, 0
#         else:
#             return index % self.training_samples, self.rand_gen.randint(0, self.size)


#     def classifier_free_diffusion(self, inputs):
#         p = self.rand_gen.random()
#         if p < self.p_dropout:
#             inputs[:] = 0
#         return inputs



# class DataIC_Vel(TrainingSetBase):
#     def __init__(self, size, n_spatial_dim, which, training_samples, learn_residual, training_variable=None, mean_training_=None, std_training_=None, start=0, aug_data=0, mc_samples=None, workdir=None, plot=False, p_dropout=0, scalar_norm=True, compute_mean=True):
#         super().__init__(size, n_spatial_dim, which, training_samples, learn_residual, training_variable, mean_training_, std_training_, start, aug_data, mc_samples, workdir, plot, p_dropout, scalar_norm)
#         self.class_name = self.__class__.__name__
#         self.input_channel = 2
#         self.output_channel = 2
#         self.stat_size = size
#         self.use_low_res_stats = False
#         self.p_dropout = p_dropout
#         # self.chunks = {'member': 1000, 'time': -1, 'c': -1, 'x': -1, 'y': -1}
#         self.ds = netCDF4.Dataset(f'/cluster/work/math/camlab-data/data/diffusion_project/ddsl_fast_{which}_{size}_tr2.nc', mode='r')


#     # @functools.lru_cache(maxsize=10000)
#     def __getitem__(self, index):

#         index += self.start

#         (index, shift) = self.get_index_shift(index) if self.start == 0 else (index, 0)
#         data = self.ds['data'][index].data

#         data_input = data[0, ..., :self.input_channel]
#         data_output = data[1, ..., :self.output_channel]

#         data_input = self.normalize_input(data_input)

#         if self.p_dropout != 0:
#             data_input = self.classifier_free_diffusion(data_input)

#         data_output = self.normalize_output(data_output)

#         inputs = np.concatenate((data_input, data_output), -1)
#         if shift != 0:
#             inputs = translate_horizontally_periodic_unbatched(inputs, shift, axis=1)
#         return inputs


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