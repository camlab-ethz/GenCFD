"""This file only has dummy dataloader!

TODO: Rewrite a Base Dataset class which can be used for the DataLoader!
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
import importlib
# from utils import check_if_time_problem, load_params_from_json
import netCDF4 as nc
import numpy as np


class DummyDataloader(Dataset):
    def __init__(self, file_path: str = None, transform: bool = None, partial: tuple = None, data: torch.Tensor = None):
        if file_path is not None:
            self.data = nc.Dataset(file_path)
        else:
            self.data = data

        self.num_data, self.time, self.x_dim, self.y_dim, self.channels = self.data['data'].shape
        self.transform = transform

        if partial is not None:
            idx_start, idx_end = partial
            self.data_tensor = torch.tensor(
                self.data['data'][idx_start:idx_end, 0, ...], 
                dtype=torch.float32
                )
        else:
            self.data_tensor = torch.tensor(
                self.data['data'],
                dtype=torch.float32
            )

    def __len__(self):
        # Number of samples in the dataset
        return self.num_data * self.time

    def __getitem__(self, idx):
        # Calculate member and time indices based on the overall index
        member_idx = idx // self.time
        time_idx = idx % self.time

        data = self.data['data'][member_idx, time_idx, :, :, :]  # Shape (x, y, c)

        if self.transform:
            data = self.transform(data)

        return torch.tensor(data, dtype=torch.float32)
    
    def get_loader(self, bs: int = 32, shuffle: bool = True, num_worker: int = 0):
        return DataLoader(
            self.data, 
            batch_size=bs, 
            shuffle=shuffle, 
            num_workers=num_worker
        )


# class BaseDataset(Dataset):
#     def __init__(self, dataset, n_spatial_dim: int):
#         self.dataset = dataset
#         self.n_spatial_dim = n_spatial_dim

#     def __len__(self):
#         return len(self.dataset)
    
#     def __getitem__(self, idx):
#         data = self.dataset[idx]
#         data = self.dataset.collate(data)
#         return data
    

# class BaseTimeDataset(Dataset):
#     def __init__(self, dataset, n_spatial_dim: int):
#         self.dataset = dataset
#         self.n_spatial_dim = n_spatial_dim

#     def __len__(self):
#         return len(self.dataset)
    
#     def __getitem__(self, idx):
#         time, data = self.dataset[idx]
#         time, data = self.dataset.collate(time, data)
#         return time, data
    

# def create_dataloader(dataset, n_spatial_dim: int, batch_size: int, shuffle: bool = True, cache: bool = False):
#     # Initialize the PyTorch Dataset
#     if check_if_time_problem(dataset.__class__.__name__):
#         pytorch_dataset = TorchTimeDataset(dataset, n_spatial_dim)
#     else:
#         pytorch_dataset = TorchDataset(dataset, n_spatial_dim)

#     # Create the DataLoader
#     dataloader = DataLoader(
#         dataset=pytorch_dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=4,  # Adjust the number of workers depending on the system
#         pin_memory=True,  # If you are using a GPU, this improves data transfer speed
#     )

#     return dataloader


# def get_loader_by_module_name(module_name, monte_carlo_samples, spatial_dim, workdir, batch_size, shuffle=False, start=0):
#     if not os.path.isdir(workdir):
#         raise NotADirectoryError(f"{workdir} not found")

#     parameters = load_params_from_json(workdir)

#     # Training parameters
#     num_training_samples = int(parameters["num_training_samples"])
#     # Data parameters
#     img_size = int(parameters["img_size"])
#     learn_residual = parameters["learn_residual"]
#     training_variable = parameters["training_variable"]
#     normalization_type = parameters["normalization_type"]
#     scalar_norm = bool(parameters["scalar_norm"])

#     # If None get the same as the training module name
#     if module_name is None:
#         module_name = parameters["module_data_name"]

#     print("Init Eval Data ", module_name)
#     EvalDataset = getattr(importlib.import_module(f"Datasets"), module_name)
#     print(spatial_dim)
#     eval_data = EvalDataset(
#         img_size,
#         spatial_dim,
#         which=normalization_type,
#         training_samples=num_training_samples,
#         mc_samples=monte_carlo_samples,
#         learn_residual=learn_residual,
#         training_variable=training_variable,
#         mean_training_=None,
#         std_training_=None,
#         start=start,
#         aug_data=0,
#         workdir=workdir,
#         scalar_norm=scalar_norm,
#         compute_mean=True
#     )

#     dataloader = create_dataloader(eval_data, n_spatial_dim=spatial_dim, batch_size=batch_size, shuffle=shuffle, cache=False)

#     return eval_data, dataloader