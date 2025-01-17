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
import pickle
import time
from typing import Optional, Union, Tuple

Tensor = torch.Tensor
Array = np.ndarray
Container = Union[Array, Tensor]

import time


def timeit(func):
    """Decorator that reports the execution time."""

    def wrapper(*args, **kwargs):
        start_time = time.time()  # Capture start time
        result = func(*args, **kwargs)  # Call the decorated function
        end_time = time.time()  # Capture end time
        print(f"{func.__name__} executed in {end_time - start_time} seconds.")
        return result

    return wrapper


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


def compute_updates(
    data: Tensor, spatial_ax: Tuple[int, ...] | int, compute_also_high_mom: bool
) -> tuple:
    """Compute energy, mean, std, max, min and spectral metrics"""
    mean_ = data.mean(dim=0)
    std_ = data.std(dim=0)
    max_ = data.amax(dim=(0,) + spatial_ax)
    min_ = data.amin(dim=(0,) + spatial_ax)

    if compute_also_high_mom:
        u_hat_ = torch.fft.fftn(data, norm="forward", dim=spatial_ax)
        kk_, sp_ = compute_spectrum(u_hat_, spatial_ax)
        sp_ = sp_.mean(dim=0)
        e_ = compute_energy(u_hat_, spatial_ax).mean(dim=0)
    else:
        kk_, sp_, e_ = None, None, None

    return (mean_, std_, min_, max_, kk_, sp_, e_)


def compute_energy(batch_u_hat_: Tensor, spatial_ax: Tuple[int, ...] | int) -> Tensor:
    return torch.sum(torch.abs(batch_u_hat_) ** 2, dim=spatial_ax)


def compute_spectrum(batch_u_hat_: Tensor, spatial_ax: Tuple[int, ...] | int):

    batch_size = batch_u_hat_.shape[0]
    channels = batch_u_hat_.shape[1]
    nf = batch_u_hat_.shape[-1]

    batch_u_hat_ = torch.fft.fftshift(
        batch_u_hat_, dim=list(range(1, batch_u_hat_.ndim - 1))
    )
    u_hat_energy = torch.abs(batch_u_hat_) ** 2

    spatial_dims = batch_u_hat_.shape[2:]
    center = [int(s / 2) for s in spatial_dims]

    if len(spatial_dims) == 2:
        box_sidex, box_sidey = spatial_dims
        px, py = torch.meshgrid(
            torch.arange(box_sidex, device=batch_u_hat_.device),
            torch.arange(box_sidey, device=batch_u_hat_.device),
            indexing="ij",
        )
        kk_matrix = torch.maximum(torch.abs(px - center[0]), torch.abs(py - center[1]))
    elif len(spatial_dims) == 3:
        box_sidex, box_sidey, box_sidez = spatial_dims
        px, py, pz = torch.meshgrid(
            torch.arange(box_sidex, device=batch_u_hat_.device),
            torch.arange(box_sidey, device=batch_u_hat_.device),
            torch.arange(box_sidez, device=batch_u_hat_.device),
            indexing="ij",
        )
        kk_matrix = torch.maximum(
            torch.maximum(torch.abs(px - center[0]), torch.abs(py - center[1])),
            torch.abs(pz - center[2]),
        )
    else:
        raise ValueError("Data must be either 2D or 3D.")

    E_u = 1e-16 + torch.zeros(
        (batch_size, nf // 2, channels), dtype=torch.float32, device=batch_u_hat_.device
    )

    for k in range(E_u.shape[1]):
        mask = kk_matrix == k
        expanded_mask = mask.unsqueeze(0).unsqueeze(-1)
        el = torch.where(
            expanded_mask,
            u_hat_energy,
            torch.tensor(0.0, dtype=torch.float32, device=batch_u_hat_.device),
        )
        el = el.sum(dim=spatial_ax)
        E_u[:, k, :] = el

    kk_vec = torch.arange(1, E_u.shape[1] + 1, device=batch_u_hat_.device)

    return kk_vec, E_u


class StatsRecorder:
    def __init__(
        self,
        idx_wassertain_: Optional[Union[int, Tuple[int, ...], Tensor]] = None,
        num_batches: int = None,
        compute_also_high_mom: bool = False,
        batch_to_keep: int = 1,
        spatial_ax: Tuple[int, ...] = (1, 2),
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ):
        self.mean = 0.0
        self.std = 0.0
        self.max = -1e16
        self.min = 1e16
        self.nobservations = 0
        self.ndimensions = None
        self.current_batch = 0
        self.spatial_ax = spatial_ax

        self.cov = 0.0
        self.energy = 0.0
        self.spectrum = 0.0
        self.kk = 0.0

        self.compute_also_high_mom = compute_also_high_mom
        self.idx_wassertain = idx_wassertain_

        self.num_batches = num_batches
        self.solution_at_idx = None
        self.batch_to_keep = batch_to_keep

        self.dtype = dtype
        self.device = device

    def update(self, data):
        """
        data: ndarray, shape (nobservations, N, N, ndimensions)
        """
        data = torch.as_tensor(data, device=self.device, dtype=self.dtype)
        self.batch_to_keep = min(self.batch_to_keep, data.shape[0])
        mean_, std_, min_, max_, kk_, sp_, e_ = compute_updates(
            data, self.spatial_ax, self.compute_also_high_mom
        )
        if self.nobservations == 0:
            self.nobservations = data.shape[0]
            self.ndimensions = data.shape[-1]

            self.mean = mean_
            self.std = std_
            self.max = max_
            self.min = min_

            if self.compute_also_high_mom:
                self.energy = e_
                self.spectrum = sp_
                self.kk = kk_
                self.solution_at_idx = torch.zeros(
                    (
                        self.num_batches,
                        self.batch_to_keep,
                        self.idx_wassertain.shape[0],
                        self.ndimensions,
                    ),
                    dtype=self.dtype,
                    device=self.device,
                )
        else:
            if (
                self.solution_at_idx is not None
                and self.num_batches >= self.solution_at_idx.shape[0]
            ):
                tmp_ = self.solution_at_idx
                old_num_batches = tmp_.shape[0]
                self.solution_at_idx = torch.zeros(
                    (
                        self.num_batches,
                        self.batch_to_keep,
                        self.idx_wassertain.shape[0],
                        self.ndimensions,
                    ),
                    dtype=self.dtype,
                    device=self.device,
                )
                self.solution_at_idx = self.solution_at_idx.at[:old_num_batches].set(
                    tmp_
                )

            if data.shape[-1] != self.ndimensions:
                raise ValueError("Data dims don't match prev observations.")
            # Compute new stats
            m = self.nobservations
            n = data.shape[0]

            tmp_mean = self.mean

            # Mean
            self.mean = m / (m + n) * tmp_mean + n / (m + n) * mean_
            # Standard Deviation
            self.std = (
                m / (m + n) * self.std**2
                + n / (m + n) * std_**2
                + m * n / (m + n) ** 2 * (tmp_mean - mean_) ** 2
            ) ** 0.5
            # Min/max
            self.min = torch.minimum(self.min, min_)
            self.max = torch.maximum(self.max, max_)

            if self.compute_also_high_mom:
                # Energy
                self.energy = m / (m + n) * self.energy + n / (m + n) * e_
                # Spectrum
                self.spectrum = m / (m + n) * self.spectrum + n / (m + n) * sp_

            self.nobservations += n

        if self.compute_also_high_mom:
            # img_size = data.shape[1]
            data_to_save = data[: self.batch_to_keep].reshape(
                self.batch_to_keep, -1, self.ndimensions
            )
            self.solution_at_idx[self.current_batch] = data_to_save[
                ..., self.idx_wassertain, :
            ]

        self.current_batch += 1

    def save_state(self, file_path):
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load_state(file_path):
        with open(file_path, "rb") as file:
            return pickle.load(file)


def downsample(u_: Tensor, N: int) -> Tensor:
    N_old = u_.shape[-2]
    freqs = torch.fft.fftfreq(N_old, d=1 / N_old)
    select_freqs = torch.logical_and(freqs >= -N / 2, freqs <= N / 2 - 1)
    u_fft = torch.fft.fft2(u_)
    u_fft_down = u_fft[..., select_freqs, :][..., select_freqs]
    u_down = torch.fft.ifft2(u_fft_down)
    return u_down


def samples_fft(u_: Tensor) -> Tensor:
    return torch.fft.fft2(u_, norm="forward")


def samples_ifft(u_hat: Tensor) -> Tensor:
    return torch.fft.ifft2(u_hat, norm="forward").real


def upsample(u: Tensor, N: int, fourier: bool = False) -> Tensor:
    # shape (batch, channel, nx, ny)
    if torch.is_floating_point(u):
        u_hat = samples_fft(u)
    elif torch.is_complex(u):
        u_hat = u
    else:
        raise TypeError(f"Expected either real or complex valued array. Got {u.dtype}.")
    dim = u_hat.ndim - 2
    N_old = u_hat.shape[-2]
    channels = u_hat.shape[1]

    if dim == 2:
        u_hat_up = torch.zeros(
            (u_hat.shape[0], channels, N, N), dtype=u_hat.dtype, device=u_hat.device
        )
        u_hat_up[..., : N_old // 2 + 1, : N_old // 2 + 1] = u_hat[
            ..., : N_old // 2 + 1, : N_old // 2 + 1
        ]
        u_hat_up[..., : N_old // 2 + 1, -N_old // 2 :] = u_hat[
            ..., : N_old // 2 + 1, -N_old // 2 :
        ]
        u_hat_up[..., -N_old // 2 :, : N_old // 2 + 1] = u_hat[
            ..., -N_old // 2 :, : N_old // 2 + 1
        ]
        u_hat_up[..., -N_old // 2 :, -N_old // 2 :] = u_hat[
            ..., -N_old // 2 :, -N_old // 2 :
        ]
    else:
        raise ValueError(f"Invalid dimension {dim}")

    if fourier:
        return u_hat_up

    u_up = samples_ifft(u_hat_up)
    return u_up


def translate_horizontally_periodic_batched(
    tensor: Tensor, pixels: int | tuple, axis: int = -1
) -> Tensor:
    """Shifts a tensor horizontally across the width (axis = -1)
    tensor has shape (bs, c, nz, ny, nx)
    """
    return torch.roll(tensor, shifts=pixels, dims=axis)


def translate_horizontally_periodic_unbatched(
    tensor: Tensor, pixels: int | tuple, axis: int = -1
) -> Tensor:
    """Shifts a tensor horizontally across the width (axis = -1)
    tensor has shape (c, nz, ny, nx)
    """
    tensor = tensor.unsqueeze(0)
    tensor = translate_horizontally_periodic_batched(tensor, pixels, axis)
    return tensor.squeeze(0)
