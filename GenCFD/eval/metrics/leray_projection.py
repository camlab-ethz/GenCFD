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

import torch
import numpy as np
from typing import Optional, Union, Tuple

Tensor = torch.Tensor

class LerayProjector:
    """
    Efficient Leray projection for 2D/3D velocity fields with precomputed operators.
    This class precomputes the projection operator once and reuses it for all batches.
    Supports both 2D (x,y) and 3D (x,y,z) configurations with anisotropic grids.
    """
    
    def __init__(
        self, 
        shape: Union[Tuple[int, int], Tuple[int, int, int]], 
        spacing: Union[float, Tuple[float, ...]] = 1.0,
        set_automatic_precision: bool = True,
        device: Optional[torch.device] = None, 
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize the Leray projector with precomputed operators.
        
        Args:
            shape: Spatial dimensions (Nx, Ny) for 2D or (Nx, Ny, Nz) for 3D
            spacing: Grid spacing - float for isotropic, tuple for anisotropic
            device: PyTorch device
            dtype: PyTorch data type
        """
        self.shape = shape[::-1]
        self.ndim = len(shape)
        self.device = device or torch.device('cpu')
        self.dtype = dtype
        
        if self.ndim not in [2, 3]:
            raise ValueError(f"Only 2D and 3D supported, got {self.ndim}D")
        
        # Handle different input formats for grid spacing
        if isinstance(spacing, (tuple, list)):
            if len(spacing) != self.ndim:
                raise ValueError(f"Spacing tuple must have {self.ndim} elements for {self.ndim}D")
            self.spacing = spacing
        else:
            self.spacing = tuple([spacing] * self.ndim)
        
        # Precision is adapated automatically for higher accuracy
        self.set_automatic_precision = set_automatic_precision
        
        # Precompute the projection operator
        self._precompute_projection_operator()
    
    def _precompute_projection_operator(self):
        """Precompute the projection operator P = I - (k ⊗ k) / |k|²"""
        
        # Create frequency grids for each dimension
        k_grids = []
        for i, (n, dx) in enumerate(zip(self.shape, self.spacing)):
            k = torch.fft.fftfreq(n, d=dx, device=self.device, dtype=self.dtype) * 2 * np.pi
            k_grids.append(k)
        
        # Create wavenumber meshgrids
        K_components = torch.meshgrid(*k_grids, indexing='ij')
        
        # Stack into wavevector K: shape (ndim, *shape)
        K = torch.stack(K_components, dim=0)
        
        # Derivative operator for divergence calculation
        self.K_hat = 1j * K
        
        # |k|² with regularization for k=0
        K_sq = (K ** 2).sum(0, keepdim=True)
        K_sq[K_sq == 0] = 1.0
        
        # Projection matrix: P = I - (k ⊗ k) / |k|²
        eye_shape = [self.ndim, self.ndim] + [1] * self.ndim
        P_real = torch.eye(self.ndim, device=self.device, dtype=self.dtype).view(eye_shape) - \
                 (K.unsqueeze(0) * K.unsqueeze(1) / K_sq)
        
        # Convert to complex for FFT operations
        complex_dtype = torch.complex64 if self.dtype == torch.float32 else torch.complex128
        self.P = P_real.to(complex_dtype)
    

    def low_pass_filter(
        self, 
        u_hat: Tensor, 
        fraction: float = 0.5
    ) -> Tensor:
        """
        Apply a low-pass filter in the frequency domain, zeroing out high-frequency modes.
        
        Args:
            u_hat: Fourier-transformed velocity field (B, C, *shape), complex
            fraction: Fraction of the lowest frequencies to retain (e.g., 0.5 retains 50%)
        
        Returns:
            Filtered u_hat
        """
        assert 0 < fraction <= 1.0, "fraction must be in (0, 1]"
        shape = u_hat.shape[2:]  # spatial shape
        dims = len(shape)
        
        # Compute the frequency grids again
        freq_grids = [
            torch.fft.fftfreq(n, d=dx, device=u_hat.device, dtype=torch.float32) * 2 * np.pi
            for n, dx in zip(self.shape, self.spacing)
        ]
        
        # Create meshgrid of squared frequency magnitude
        K_components = torch.meshgrid(*freq_grids, indexing='ij')
        k_squared = sum(k**2 for k in K_components)

        # Flatten and sort
        k_flat = k_squared.flatten()
        k_thresh = torch.quantile(k_flat, fraction)

        # Build mask of low frequencies
        mask = k_squared <= k_thresh  # shape (*shape,)

        # Reshape for broadcasting
        mask = mask[None, None, ...]  # shape (1, 1, *shape)
        return u_hat * mask

    
    def fft_filter(
        self,
        u_batch: Tensor,
        overwrite_automatic_precision: bool = False,
        filter_fraction: float = 0.5
    ) -> Tensor:
        """
        Apply a low-pass FFT filter on the velocity field without projection.
        
        Args:
            u_batch: Velocity field in physical space (B, C, *spatial_shape)
            filter_fraction: Fraction of lowest frequencies to keep, e.g. 0.5 keeps 50%
        
        Returns:
            Filtered velocity field in physical space, same shape as input.
        """
        # Move to correct device and dtype
        u_batch = u_batch.to(device=self.device, dtype=self.dtype)

        if (
            # self.device == 'cuda' and 
            (self.set_automatic_precision or overwrite_automatic_precision)
        ):
            # Use higher precision FFT if set (similar to project)
            u_batch_compute = u_batch.to(torch.float64)
        else:
            u_batch_compute = u_batch

        fft_dims = tuple(range(-self.ndim, 0))
        u_hat = torch.fft.fftn(u_batch_compute, dim=fft_dims)

        # Apply your existing low_pass_filter method
        u_hat_filtered = self.low_pass_filter(u_hat, fraction=filter_fraction)

        # Inverse FFT back to physical space
        u_filtered = torch.fft.ifftn(u_hat_filtered, dim=fft_dims).real

        # Return to original dtype
        return u_filtered.to(self.dtype)


    def project(
        self, 
        u_batch: Tensor,
        overwrite_automatic_precision: bool = False,
        filter_fraction: Optional[float] = None
    ) -> Tensor:
        """
        Apply Leray projection to a batch of velocity fields.
    
        Args:
            u_batch: Shape (B, ch, *spatial_resolution) - batch of velocity fields
            overwrite_automatic_precision: if activated FFT is handled more 
                carefully with higher precision.
        
        Returns:
            torch.Tensor: Projected velocity fields
        """
        # Ensure input is on the same device as the projector
        u_batch = u_batch.to(device=self.device, dtype=self.dtype)

        if (
            # self.device == 'cuda' and 
            (self.set_automatic_precision or overwrite_automatic_precision)
        ):
            # Use double precision for GPU FFT operations to match CPU precision
            u_batch_compute = u_batch.to(torch.float64)
            P_compute = self.P.to(torch.complex128)
        else:
            u_batch_compute = u_batch
            P_compute = self.P
    
        fft_dims = tuple(range(-self.ndim, 0))
        u_hat = torch.fft.fftn(u_batch_compute, dim=fft_dims)
    
        if self.ndim == 2:
            u_hat_proj = torch.einsum('ijxy,bjxy->bixy', P_compute, u_hat)
        else:  # 3D
            u_hat_proj = torch.einsum('ijxyz,bjxyz->bixyz', P_compute, u_hat)
        
        if filter_fraction is not None:
            u_hat_proj = self.low_pass_filter(u_hat_proj, fraction=filter_fraction)
    
        # Return to physical space
        u_proj = torch.fft.ifftn(u_hat_proj, dim=fft_dims).real
    
        # Convert back to original dtype
        return u_proj.to(self.dtype)


    def calculate_divergence(
        self, 
        u_batch: Tensor,
        overwrite_automatic_precision: bool = False
    ) -> Tensor:
        """Calculate divergence of velocity field with consistent precision."""
        # Ensure input is on the same device as the projector
        u_batch = u_batch.to(device=self.device, dtype=self.dtype)
        
        if (
            # self.device == 'cuda' and 
            (self.set_automatic_precision or overwrite_automatic_precision)
        ):
            # Use double precision for GPU FFT operations
            u_batch_compute = u_batch.to(torch.float64)
            K_hat_compute = self.K_hat.to(torch.complex128)
        else:
            u_batch_compute = u_batch
            K_hat_compute = self.K_hat
        
        fft_dims = tuple(range(-self.ndim, 0))
        u_hat = torch.fft.fftn(u_batch_compute, dim=fft_dims)
        
        # Calculate divergence in frequency domain
        divergence_hat = (u_hat * K_hat_compute.unsqueeze(0)).sum(1)
        divergence = torch.fft.ifftn(divergence_hat, dim=fft_dims).real
        
        # Convert back to original dtype
        return divergence.to(self.dtype)


    def calculate_L2_divergence(
        self, 
        u_batch: Tensor, 
        overwrite_automatic_precision: bool = False
    ) -> Union[Tensor, float]:
        """
        Calculate L2 norm of divergence of the input velocity field.
        
        Args:
            u_batch: Shape (B, ch, *spatial_resolution) - batch of velocity fields
            
        Returns:
            torch.Tensor or float: L2 norm of the divergence field
        """
        # Ensure input is on the same device as the projector
        # u_batch = u_batch.to(device=self.device, dtype=self.dtype)
        
        # Use double precision for GPU FFT operations to match CPU precision
        
        if (
            # self.device == 'cuda' and 
            (self.set_automatic_precision or overwrite_automatic_precision)
        ):
            u_batch_compute = u_batch.to(torch.float64)
            K_hat_compute = self.K_hat.to(torch.complex128)
        else:
            u_batch_compute = u_batch
            K_hat_compute = self.K_hat

        
        fft_dims = tuple(range(-self.ndim, 0))
        u_hat = torch.fft.fftn(u_batch_compute, dim=fft_dims)
        
        # Calculate divergence directly (no projection!)
        divergence_hat = (u_hat * K_hat_compute.unsqueeze(0)).sum(1)
        divergence = torch.fft.ifftn(divergence_hat, dim=fft_dims).real
        
        # Calculate L2 norm
        l2_norm = torch.sqrt(torch.mean(divergence**2, dim=fft_dims))
        return l2_norm.mean().to(self.dtype)


    def calculate_projected_divergence(
        self, 
        u_batch: Tensor,
        overwrite_automatic_precision: bool = False
    ) -> Tensor:
        """Calculate divergence of the projected velocity field."""
        # Ensure input is on the same device as the projector
        u_batch = u_batch.to(device=self.device, dtype=self.dtype)
        
        # Use double precision for GPU FFT operations to match CPU precision
        if (
            # self.device == 'cuda' and 
            (self.set_automatic_precision or overwrite_automatic_precision)
        ):
            u_batch_compute = u_batch.to(torch.float64)
            P_compute = self.P.to(torch.complex128)
            K_hat_compute = self.K_hat.to(torch.complex128)
        else:
            u_batch_compute = u_batch
            P_compute = self.P
            K_hat_compute = self.K_hat
        
        fft_dims = tuple(range(-self.ndim, 0))
        u_hat = torch.fft.fftn(u_batch_compute, dim=fft_dims)
        
        if self.ndim == 2:
            u_hat_proj = torch.einsum('ijxy,bjxy->bixy', P_compute, u_hat)
        else:  # 3D
            u_hat_proj = torch.einsum('ijxyz,bjxyz->bixyz', P_compute, u_hat)
            
        divergence_hat = (u_hat_proj * K_hat_compute.unsqueeze(0)).sum(1)
        divergence = torch.fft.ifftn(divergence_hat, dim=fft_dims).real
        
        # Convert back to original dtype
        return divergence.to(self.dtype)
