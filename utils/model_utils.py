import torch
import torch.nn as nn
import numpy as np
from typing import Sequence
from utils.parser_utils import ArgumentParser

from diffusion.diffusion import (
    NoiseLevelSampling, 
    NoiseLossWeighting
)

Tensor = torch.Tensor

def reshape_jax_torch(tensor: Tensor) -> Tensor:
    """
    A jax based dataloader is off shape (bs, width, height, depth, c),
    while a PyTorch based dataloader is off shape (bs, c, depth, height, width).

    It transforms a tensor for the 2D and 3D case as follows:
    - 2D: (bs, c, depth, height, width) <-> (bs, width, height, depth, c)
    - 3D: (bs, c, height, width) <-> (bs, width, height, c)
    """
    if len(tensor.shape) == 3:
       # Reshape for the 1D case:
       return tensor.permute(0, 2, 1)
    elif len(tensor.shape) == 4:
        # Reshape for the 2D case
        return tensor.permute(0, 3, 2, 1)
    elif len(tensor.shape) == 5:
        # Reshape for the 3D case
        return tensor.permute(0, 4, 3, 2, 1)
    else:
        raise ValueError(
            f"Incorrect tensor shape, only 3D, 4D and 5D tensors are valid"
            )


def default_init(scale: float = 1e-10):
  """Initialization of weights and biases with scaling"""

  def initializer(tensor: Tensor):
    """We need to differentiate between biases and weights"""

    if tensor.ndim == 1: # if bias
       bound = torch.sqrt(torch.tensor(3.0)) * scale
       with torch.no_grad():
          return tensor.uniform_(-bound, bound)
       
    else: # if weights
      fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
      std = torch.sqrt(torch.tensor(scale / ((fan_in + fan_out) / 2.0)))
      bound = torch.sqrt(torch.tensor(3.0)) * std # uniform dist. scaling factor
      with torch.no_grad():
        return tensor.uniform_(-bound, bound)
        
  return initializer


def channel_to_space(inputs: Tensor, block_shape: Sequence[int]) -> Tensor:
  """Reshapes data from the channel to spatial dims as a way to upsample.

  As an example, for an input of shape (*batch, x, y, z) and block_shape of
  (a, b), additional spatial dimensions are first formed from the channel
  dimension (always the last one), i.e. reshaped into
  (*batch, x, y, a, b, z//(a*b)). Then the new axes are interleaved with the
  original ones to arrive at shape (*batch, x, a, y, b, z//(a*b)). Finally, the
  new axes are merged with the original axes to yield final shape
  (*batch, x*a, y*b, z//(a*b)).

  Args:
    inputs: The input array to upsample.
    block_shape: The shape of the block that will be formed from the channel
      dimension. The number of elements (i.e. prod(block_shape) must divide the
      number of channels).

  Returns:
    The upsampled array.
  """
  # reshape from (bs, c, y, x) to (bs, x, y, c)
  inputs = reshape_jax_torch(inputs)
  if not len(inputs.shape) > len(block_shape):
    raise ValueError(
        f"Ndim of `x` ({len(inputs.shape)}) expected to be higher than the length of"
        f" `block_shape` {len(block_shape)}."
    )

  if inputs.shape[-1] % np.prod(block_shape) != 0:
    raise ValueError(
        f"The number of channels in the input ({inputs.shape[-1]}) must be"
        f" divisible by the block size ({np.prod(block_shape)})."
    )

  # Create additional spatial axes from channel.
  old_shape = inputs.shape
  batch_ndim = inputs.ndim - len(block_shape) - 1
  cout = old_shape[-1] // np.prod(block_shape)
  x = torch.reshape(inputs, old_shape[:-1] + tuple(block_shape) + (cout,))

  # Interleave old and new spatial axes.
  spatial_axes = np.arange(2 * len(block_shape), dtype=np.int32) + batch_ndim
  new_axes = spatial_axes.reshape(2, -1).ravel(order="F")
  x = torch.permute(
      x,
      tuple(range(batch_ndim))
      + tuple(new_axes)
      + (len(new_axes) + batch_ndim,),
  )

  # Merge the interleaved axes.
  new_shape = np.asarray(old_shape[batch_ndim:-1]) * np.asarray(block_shape)
  new_shape = old_shape[:batch_ndim] + tuple(new_shape) + (cout,)
  return reshape_jax_torch(torch.reshape(x, new_shape))


def get_model_args(
    args: ArgumentParser, 
    out_channels: int, 
    rng: torch.Generator, 
    device: torch.device = None,
    mean_training_input: Tensor = None,
    mean_training_output: Tensor = None,
    std_training_input: Tensor = None,
    std_training_output: Tensor = None
  ) -> dict:
  """Return a dictionary of model parameters for the UNet architecture"""

  if args.model_type in ['UNet', 'PreconditionedDenoiser']:
    # General UNet arguments for the 2D case
    args_dict_2d = {
      'out_channels': out_channels, 'rng': rng, 'resize_to_shape': args.resize_to_shape,
      'use_hr_residual': args.use_hr_residual, 'num_channels': args.num_channels, 
      'downsample_ratio': args.downsample_ratio, 'num_blocks': args.num_blocks,
      'noise_embed_dim': args.noise_embed_dim, 'padding_method': args.padding_method,
      'dropout_rate': args.dropout_rate, 'use_attention': args.use_attention, 
      'use_position_encoding': args.use_position_encoding, 'num_heads': args.num_heads,
      'normalize_qk': args.normalize_qk, 'device': device, 
      'mean_training_input': mean_training_input, 'mean_training_output': mean_training_output, 
      'std_training_input': std_training_input, 'std_training_output': std_training_output
    }
    if args.model_type == 'PreconditionedDenoiser':
      args_dict_2d.update({'sigma_data': args.sigma_data})

    args_dict = args_dict_2d
  
  if args.model_type in ['UNet3D', 'PreconditionedDenoiser3D']:
    # General UNet arguments for the 3D case
    args_dict_3d = {
      'out_channels': out_channels, 'rng': rng, 'num_channels': (64, 128, 256), 
      'downsample_ratio': (2, 2, 2), 'num_blocks': args.num_blocks,
      'noise_embed_dim': args.noise_embed_dim, 'input_proj_channels': args.noise_embed_dim,
      'output_proj_channels': args.noise_embed_dim, 'padding_method': args.padding_method,
      'dropout_rate': args.dropout_rate, 'use_spatial_attention': (True, True, True),
      'use_position_encoding': args.use_position_encoding, 'num_heads': args.num_heads,
      'normalize_qk': args.normalize_qk, 'dtype': args.dtype, 'device': device,
      'mean_training_input': mean_training_input, 'mean_training_output': mean_training_output, 
      'std_training_input': std_training_input, 'std_training_output': std_training_output
    }
    if args.model_type == 'PreconditionedDenoiser3D':
      args_dict_3d.update({'sigma_data': args.sigma_data})
    
    args_dict = args_dict_3d
      
  return args_dict

# General Denoiser arguments
def get_denoiser_args(
    args: ArgumentParser,
    input_shape: int,
    input_channels: int,
    denoiser: nn.Module,
    noise_sampling: NoiseLevelSampling,
    noise_weighting: NoiseLossWeighting,
    rng: torch.Generator,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32
  ) -> dict:
  """Return a dictionary of parameters for the DenoisingModel"""

  denoiser_args = {
    'input_shape': input_shape, 'denoiser': denoiser, 'noise_sampling': noise_sampling,
    'noise_weighting': noise_weighting, 'rng': rng, 
    'num_eval_noise_levels': args.num_eval_noise_levels, 
    'num_eval_cases_per_lvl': args.num_eval_cases_per_lvl,
    'min_eval_noise_lvl': args.min_eval_noise_lvl, 'max_eval_noise_lvl': args.max_eval_noise_lvl,
    'consistent_weight': args.consistent_weight, 'device': device, 'dtype': dtype
  }

  if args.unconditional:
     return denoiser_args
  
  denoiser_args.update({'input_channel': input_channels, 'task': args.task})

  return denoiser_args