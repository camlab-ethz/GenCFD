# Copyright 2024 The swirl_dynamics Authors.
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

"""Generic model class for use in gradient descent mini-batch training."""

import dataclasses
from typing import (
    Any, 
    Protocol, 
    Optional, 
    Mapping,
    Callable,
    Union, 
    Tuple
)
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics import MetricCollection
import numpy as np
import diffusion as dfn_lib
from diffusion import samplers

from solvers.sde import EulerMaruyama

# from GPUtil.GPUtil import getGPUs


Tensor = torch.Tensor
TensorDict = Mapping[str, Tensor]
BatchType = Mapping[str, Union[np.ndarray, Tensor]]
ModelVariable = Union[dict, tuple[dict, ...], Mapping[str, dict]]
PyTree = Any
LossAndAux = tuple[Tensor, tuple[TensorDict, PyTree]]
Metrics = dict # Placeholder for metrics that are implemented!


class BaseModel(ABC):
  """Base class for models.

  Wraps flax module(s) to provide interfaces for variable
  initialization, computing loss and evaluation metrics. These interfaces are
  to be used by a trainer to perform gradient updates as it steps through the
  batches of a dataset.

  Subclasses must implement the abstract methods.
  """

  @abstractmethod
  def initialize(self) -> ModelVariable:
    """Initializes variables of the wrapped flax module(s).

    This method by design does not take any sample input in its argument. Input
    shapes are expected to be statically known and used to create
    initialization input for the model. For example::

      import torch.nn as nn
      
      class MLP(BaseModel):
        def __init__(self, input_shape: tuple[int], hidden_size: int):
          super().__init__()
          self.model = nn.Sequential(
            nn.Linear(np.prod(input_shape), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, np.pord(input_shape))
          )
          self.input_shape = input_shape

    Returns:
      The initial variables for this model - can be a single or a tuple/mapping
      of PyTorch variables.
    """
    raise NotImplementedError

  @abstractmethod
  def loss_fn(
      self,
      params: Union[PyTree, tuple[PyTree, ...]],
      batch: BatchType,
      rng: torch.Generator,
      mutables: PyTree,
      **kwargs,
  ) -> LossAndAux:
    """Computes training loss and metrics.

    It is expected that gradient would be taken (via `jax.grad`) wrt `params`
    during training.

    Arguments:
      params: model parameters wrt which the loss would be differentiated.
      batch: a single batch of data.
      rng: jax random key for randomized loss if needed.
      mutables: model variables which are not differentiated against; can be
        mutable if so desired.
      **kwargs: additional static configs.

    Returns:
      loss: the (scalar) loss function value.
      aux: two-item auxiliary data consisting of
        metric_vars: a dict with values required for metric compute and logging.
          They can either be final metric values computed inside the function or
          intermediate values to be further processed into metrics.
        mutables: non-differentiated model variables whose values may change
          during function execution (e.g. batch stats).
    """
    raise NotImplementedError

  def eval_fn(
      self,
      variables: Union[tuple[PyTree, ...], PyTree],
      batch: BatchType,
      rng: torch.Generator,
      **kwargs,
  ) -> TensorDict:
    """Computes evaluation metrics."""
    raise NotImplementedError

  @staticmethod
  def inference_fn(variables: PyTree, **kwargs) -> Callable[..., Any]:
    """Returns an inference function with bound variables."""
    raise NotImplementedError


"""Training a denoising model for diffusion-based generation."""


class DenoisingTorchModule(Protocol):
  """Expected interface of the flax module compatible with `DenoisingModel`.
  For the PyTorch based version we don't need to worry about that!

  NOTE: This protocol is for reference only and not statically checked.
  """

  def forward(
      self, x: Tensor, y: Tensor, time: Tensor, sigma: Tensor, is_training: bool
      ) -> Tensor:
    ...


@dataclasses.dataclass(frozen=True, kw_only=True)
class DenoisingBaseModel(BaseModel):
  """Trains a model to remove Gaussian noise from samples.

  Attributes:
    input_shape: Shape of a single sample (excluding any batch dimensions).
    denoiser: The flax module for denoising. Its `__call__` method should adhere
      to the `DenoisingFlaxModule` interface.
    noise_sampling: Callable for generating noise levels during training.
    noise_weighting: Callable for calculating loss weights based on noise
      levels.
    num_eval_cases_per_lvl: Number of evaluation samples created per noise
      level. These are generated by adding noise to random members of the
      evaluation batch.
    min_eval_noise_lvl: Minimum noise level used during evaluation.
    max_eval_noise_lvl: Maximum noise level used during evaluation.
    num_eval_noise_levels: Number of noise levels for evaluation (log-uniformly
      spaced between the minimum and maximum).
    consistent_weight: weighting for some loss terms
    device: necessary to specify whether gpu or cpu is used
    dtype: sets the precision for the type torch float
  """

  input_shape: tuple[int, ...]
  denoiser: nn.Module
  noise_sampling: dfn_lib.NoiseLevelSampling
  noise_weighting: dfn_lib.NoiseLossWeighting
  rng: torch.Generator
  num_eval_noise_levels: int = 5
  num_eval_cases_per_lvl: int = 1
  min_eval_noise_lvl: float = 1e-3
  max_eval_noise_lvl: float = 50.0
  
  consistent_weight: float = 0
  device: Any | None = None
  dtype: torch.dtype = torch.float32


  def initialize(self, batch_size: int):
    """Method necessary for a dummy initialization!"""
    x_sample = torch.ones((batch_size,) + self.input_shape, dtype=self.dtype, device=self.device)
    return self.denoiser(
        x=x_sample, sigma=torch.ones((batch_size,), dtype=self.dtype, device=self.device), is_training=False
    )

  def loss_fn(
      self,
      batch: dict,
      mutables: Optional[dict] = None
  ):
    """Computes the denoising loss on a training batch.

    Args:
      batch: A batch of training data expected to contain an `x` field with a
        shape of `(batch, *spatial_dims, channels)`, representing the unnoised
        samples. Optionally, it may also contain a `cond` field, which is a
        dictionary of conditional inputs.
      mutables: The mutable (non-diffenretiated) parameters of the denoising
        model (e.g. batch stats); *currently assumed empty*.

    Returns:
      The loss value and a tuple of training metric and mutables.
    """
    data = batch

    batch_size = len(data)
    x_squared = torch.square(data)

    sigma = self.noise_sampling(rng=self.rng, shape=(batch_size,))
    weights = self.noise_weighting(sigma)
    if weights.ndim != data.ndim:
      weights = weights.view(-1, *([1] * (data.ndim - 1)))

    noise = torch.randn(
      data.shape, dtype=self.dtype, device=self.device, generator=self.rng
      )

    if sigma.ndim != data.ndim:
      noised = data + noise * sigma.view(-1, *([1] * (data.ndim - 1))) 
    else:
      noised = data + noise * sigma
    
    denoised = self.denoiser.forward(x=noised, sigma=sigma, is_training=True)
    denoised_square = torch.square(denoised)

    rel_norm = torch.mean(torch.square(data) / torch.mean(torch.square(x_squared)))
    loss = torch.mean(weights * torch.square(denoised - data))
    loss += self.consistent_weight * rel_norm * \
            torch.mean(weights * torch.square(denoised_square - x_squared))
    
    metric = {
      "loss": loss.item(),
      "mem": 0. # TODO: Placeholder for memory metric!
    }

    return loss, (metric, mutables)
  

  def eval_fn(
      self,
      batch: dict
  ):
    """Compute denoising metrics on an eval batch.

    Randomly selects members of the batch and noise them to a number of fixed
    levels. Each level is aggregated in terms of the average L2 error.

    Args:
      batch: A batch of evaluation data expected to contain an `x` field with a
        shape of `(batch, *spatial_dims, channels)`, representing the unnoised
        samples. Optionally, it may also contain a `cond` field, which is a
        dictionary of conditional inputs.

    Returns:
      A dictionary of denoising-based evaluation metrics.
    """
    # time = batch["lead_time"]
    # data = batch["data"]
    data = batch
    inputs = data[
      torch.randint(0, data.shape[0], (self.num_eval_noise_levels, self.num_eval_cases_per_lvl), 
                    generator=self.rng, device=self.device)
      ]

    sigma = torch.exp(
      torch.linspace(
        np.log(self.min_eval_noise_lvl),
        np.log(self.max_eval_noise_lvl),
        self.num_eval_noise_levels, 
        dtype=self.dtype,
        device=self.device
      )
    )

    noise = torch.randn(
      inputs.shape, device=self.device, dtype=self.dtype, generator=self.rng
      )
    
    if sigma.ndim != inputs.ndim:
      noised = inputs + noise * sigma.view(-1, *([1] * (inputs.ndim - 1))) 
    else:
      noised = inputs + noise * sigma

    inference_function = self.inference_fn(self.denoiser, self.task)
    denoised = torch.stack(
      [inference_function(noised[i], sigma[i]) for i in range(self.num_eval_noise_levels)]
    )

    ema_losses = torch.mean(torch.square(denoised - inputs), dim=[i for i in range(1, inputs.ndim)])

    eval_losses = {f"eval_denoise_lvl{i}": loss.item() for i, loss in enumerate(ema_losses)}
    return eval_losses


  @staticmethod
  def inference_fn(denoiser: nn.Module) -> Tensor:
    """Returns the inference denoising function."""

    def _denoise(
        x: Tensor, sigma: float | Tensor, cond: Mapping[str, Tensor] | None = None
      ) -> Tensor:
    
      if not torch.is_tensor(sigma):
        sigma = sigma * torch.ones((x.shape[0],))
  
      return denoiser.forward(x=x, sigma=sigma, is_training=False)
  
    return _denoise
  




@dataclasses.dataclass(frozen=True, kw_only=True)
class DenoisingModel(DenoisingBaseModel):
  """Trains a model to remove Gaussian noise from samples.

  Additional Attributes:
    denoiser: The flax module for denoising. Its `__call__` method should adhere
      to the `DenoisingFlaxModule` interface.
  """
  
  input_channel: int = 1
  task: str = 'solver'
  # tspan_method: str = 'exponential_noise_decay'
  # compute_crps: bool = False


  def initialize(self, batch_size: int, time_cond: bool = False):
    """Method necessary for a dummy initialization!"""
    x_sample = torch.ones((batch_size,) + self.input_shape, dtype=self.dtype, device=self.device)
    x = x_sample[:, self.input_channel:, ...]
    y = x_sample[:, :self.input_channel, ...]

    if time_cond:
      time = torch.ones((batch_size,), dtype=self.dtype, device=self.device)
      return self.denoiser(
        x=x,
        y=y,
        sigma=torch.ones((batch_size,), dtype=self.dtype, device=self.device), 
        time=time,
        is_training=False
      )
    else:
      return self.denoiser(
          x=x, 
          y=y, 
          sigma=torch.ones((batch_size,), dtype=self.dtype, device=self.device), 
          is_training=False
      )

  def loss_fn(
      self,
      batch: dict,
      mutables: Optional[dict] = None
  ):
    """Computes the denoising loss on a training batch.

    Args:
      batch: A batch of training data expected to contain an `x` field with a
        shape of `(batch, channels, *spatial_dims)`, representing the unnoised
        samples. Optionally, it may also contain a `cond` field, which is a
        dictionary of conditional inputs.
      mutables: The mutable (non-diffenretiated) parameters of the denoising
        model (e.g. batch stats); *currently assumed empty*.

    Returns:
      The loss value and a tuple of training metric and mutables.
    """
    if isinstance(batch, list):
      time = batch[0]
      data = batch[1]
    else:
      data = batch
      time = None

    x = data[:, self.input_channel:, ...] # Input
    y = data[:, :self.input_channel, ...] # Output

    batch_size = len(data)
    x_squared = torch.square(x)

    sigma = self.noise_sampling(rng=self.rng, shape=(batch_size,))

    weights = self.noise_weighting(sigma)
    if weights.ndim != data.ndim:
      weights = weights.view(-1, *([1] * (data.ndim - 1)))

    noise = torch.randn(
      x.shape, dtype=self.dtype, device=self.device, generator=self.rng
    )

    if sigma.ndim != data.ndim:
      noised = x + noise * sigma.view(-1, *([1] * (data.ndim - 1))) 
    else:
      noised = x + noise * sigma
    
    if time is not None:
      denoised = self.denoiser.forward(x=noised, y=y, sigma=sigma, time=time, is_training=True)
    else:
      denoised = self.denoiser.forward(x=noised, y=y, sigma=sigma, is_training=True)

    denoised_squared = torch.square(denoised)

    rel_norm = torch.mean(torch.square(x) / torch.mean(torch.square(x_squared)))
    loss = torch.mean(weights * torch.square(denoised - x))
    loss += self.consistent_weight * rel_norm * \
            torch.mean(weights * torch.square(denoised_squared - x_squared))
    
    metric = {
      "loss": loss.item(),
      "mem": 0. # TODO: Placeholder for memory metric!
    }

    return loss, (metric, mutables)
  

  def eval_fn(
      self,
      batch: dict
  ) -> dict:
    """Compute denoising metrics on an eval batch.

    Randomly selects members of the batch and noise them to a number of fixed
    levels. Each level is aggregated in terms of the average L2 error.

    Args:
      variables: Variables for the denoising module.
      batch: A batch of evaluation data expected to contain an `x` field with a
        shape of `(batch, *spatial_dims, channels)`, representing the unnoised
        samples. Optionally, it may also contain a `cond` field, which is a
        dictionary of conditional inputs.
      rng: Random key for evaluation use.

    Returns:
      A dictionary of denoising-based evaluation metrics.
    """
    if isinstance(batch, list):
      time = batch[0]
      data = batch[1]
    else:
      data = batch
      time = None

    rand_idx_set = torch.randint(
      0, data.shape[0], 
      (self.num_eval_noise_levels, self.num_eval_cases_per_lvl), 
      generator=self.rng, device=self.device
    )

    inputs = data[rand_idx_set]

    if time is not None:
      time_inputs = time[rand_idx_set]

    sigma = torch.exp(
      torch.linspace(
        np.log(self.min_eval_noise_lvl),
        np.log(self.max_eval_noise_lvl),
        self.num_eval_noise_levels, 
        dtype=self.dtype,
        device=self.device
      )
    )

    x = inputs[:, :, self.input_channel:, ...]
    y = inputs[:, :, :self.input_channel, ...]

    noise = torch.randn(
      x.shape, device=self.device, dtype=self.dtype, generator=self.rng
      )
    
    if sigma.ndim != inputs.ndim:
      noised = x + noise * sigma.view(-1, *([1] * (inputs.ndim - 1))) 
    else:
      noised = x + noise * sigma

    denoise_fn = self.inference_fn(denoiser=self.denoiser, task=self.task, lead_time=False if time is None else True)

    if time is not None:
      denoised = torch.stack(
        [denoise_fn(x=noised[i], y=y[i], sigma=sigma[i], time=time_inputs[i]) for i in range(self.num_eval_noise_levels)]
      )
    else:
      denoised = torch.stack(
        [denoise_fn(x=noised[i], y=y[i], sigma=sigma[i]) for i in range(self.num_eval_noise_levels)]
      )

    ema_losses = torch.mean(torch.square(denoised - x), dim=[i for i in range(1, inputs.ndim)])
    eval_losses = {f"eval_denoise_lvl{i}": loss.item() for i, loss in enumerate(ema_losses)}
    return eval_losses


  @staticmethod
  def inference_fn(denoiser: nn.Module, task: str = 'solver', lead_time: bool = False) -> Tensor:
    """Returns the inference denoising function.
    Args:
      denoiser: Neural Network (NN) Module for the forward pass
      task: defines what the NN model should be used for as an N to N model 
        or as a superresolver. Where the superresolver task setting can also
        be used for a model which runs without any conditioning
      lead_time: If set to True it can be used for datasets which have time 
        included. This time value can then be used for conditioning. Commonly
        done for an All2All training strategy.
    
    Return:
      _denoise: corresponding denoise function
    """
    
    if task == 'superresolver':
      def _denoise(
          x: Tensor, sigma: float | Tensor, cond: Mapping[str, Tensor] | None = None
        ) -> Tensor:
      
        if not torch.is_tensor(sigma):
          sigma = sigma * torch.ones((x.shape[0],))
    
        return denoiser.forward(x=x, sigma=sigma, is_training=False)
      
    elif task == 'solver' and lead_time == False:
      def _denoise(
          x: Tensor, 
          sigma: float | Tensor, 
          y: Tensor,
          cond: Mapping[str, Tensor] | None = None
        ) -> Tensor:
      
        if not torch.is_tensor(sigma):
          sigma = sigma * torch.ones((x.shape[0],))
    
        return denoiser.forward(x=x, sigma=sigma, y=y, is_training=False)
    
    elif task == 'solver' and lead_time == True:
      def _denoise(
          x: Tensor, 
          sigma: float | Tensor, 
          y: Tensor,
          time: float | Tensor,
          cond: Mapping[str, Tensor] | None = None
        ) -> Tensor:
      
        if not torch.is_tensor(sigma):
          sigma = sigma * torch.ones((x.shape[0],))

        if not torch.is_tensor(time):
          time = time * torch.ones((x.shape[0],))
    
        return denoiser.forward(x=x, sigma=sigma, y=y, time=time, is_training=False)
      
    else:
      raise ValueError("model can either be used as a 'superresolver' or a 'solver'")
  
    return _denoise


  # def inference_loop(
  #     self,
  #     batch: dict,
  #     eval_losses: dict
  # ) -> dict:
  #   """Compute further generated samples and do an inference run for further evaluation

  #   Further Args compared to eval_fn:
  #     eval_losses: Dictionary with relevant metrics to collect

  #   Returns:
  #     A dictionary with the complete evaluation metrics loop
  #   """
  #   denoise_fn = self.inference_fn(self.denoiser)

  #   x_gen = batch[:, self.input_channel:, ...]
  #   y_gen = batch[:, :self.input_channel, ...]

  #   batch_size = len(batch)

  #   output_channels = self.input_shape[0] - self.input_channel # input shape is given without the batch size
  #   spatial_shape = self.input_shape[1:]
  #   spatial_axis = [i for i in range(1, len(spatial_shape) + 1)]

  #   shape = (batch_size,) + output_channels + spatial_shape
  #   tspan = getattr(samplers, self.tspan_method)(scheme=self.diffusion_scheme, num_steps=128, end_sigma=1e-4)
  #   sampler = dfn_lib.SdeSampler(
  #     input_shape=shape[1:],
  #     integrator=EulerMaruyama(self.rng),
  #     scheme=self.diffusion_scheme,
  #     denoise_fn=denoise_fn,
  #     tspan=tspan,
  #     apply_denoise_at_end=True,
  #     return_full_paths=False,
  #     rng=self.rng,
  #     device=self.dtype
  #   )

  #   cond_samples = 4
  #   x_cond = torch.zeros((cond_samples,) + shape, device=self.device, dtype=self.dtype)
  #   for i in range(cond_samples):
  #     x_cond[i] = sampler.generate(num_samples=batch_size, y=y_gen) 

  #   conditional_mean = torch.mean(x_cond, dim=0)
  #   conditional_std = torch.std(x_cond, dim=0)

  #   relative_norm_crps_score = 0
  #   relative_validation_error = 0

  #   for l in range(output_channels):
  #     crps_score_l = None # TODO!!!!
  #     norm_crps_score_l = None # TODO!!!
    
  #   # TODO: Implement crps!