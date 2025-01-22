# Copyright 2024 The swirl_dynamics Authors.
# Modifications made by the CAM Lab at ETH Zurich.
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
from typing import Any, Optional, Mapping, Callable, Union, Sequence
from abc import ABC, abstractmethod
import torch
import torch.profiler
import torch.nn as nn
import numpy as np
import GenCFD.diffusion as dfn_lib

Tensor = torch.Tensor
TensorDict = Mapping[str, Tensor]
BatchType = Mapping[str, Union[np.ndarray, Tensor]]
ModelVariable = Union[dict, tuple[dict, ...], Mapping[str, dict]]
PyTree = Any
LossAndAux = tuple[Tensor, tuple[TensorDict, PyTree]]
Metrics = dict  # Placeholder for metrics that are implemented!


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
        mutables: PyTree,
        **kwargs,
    ) -> LossAndAux:
        """Computes training loss and metrics.

        It is expected that gradient would be taken (via `jax.grad`) wrt `params`
        during training. This can also be required if via torch autograd is used!

        Arguments:
          params: model parameters wrt which the loss would be differentiated.
          batch: a single batch of data.
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
        **kwargs,
    ) -> TensorDict:
        """Computes evaluation metrics."""
        raise NotImplementedError

    @staticmethod
    def inference_fn(variables: PyTree, **kwargs) -> Callable[..., Any]:
        """Returns an inference function with bound variables."""
        raise NotImplementedError


"""Training a denoising model for diffusion-based generation."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class DenoisingModel(BaseModel):
    """Trains a model to remove Gaussian noise from samples.

    Additional Attributes:
      denoiser: The flax module for denoising. Its `__call__` method should adhere
        to the `DenoisingFlaxModule` interface.
    """

    spatial_resolution: Sequence[int]
    denoiser: nn.Module
    noise_sampling: dfn_lib.NoiseLevelSampling
    noise_weighting: dfn_lib.NoiseLossWeighting
    num_eval_noise_levels: int = 5
    num_eval_cases_per_lvl: int = 1
    min_eval_noise_lvl: float = 1e-3
    max_eval_noise_lvl: float = 50.0

    consistent_weight: float = 0
    device: Any | None = None
    dtype: torch.dtype = torch.float32

    task: str = "solver"
    time_cond: bool = False

    def initialize(
        self,
        batch_size: int,
        time_cond: bool = False,
        input_channels: int = 1,
        output_channels: int = 1,
    ):
        """Method necessary for a dummy initialization!"""

        x = torch.ones(
            (batch_size,) + (output_channels,) + self.spatial_resolution,
            dtype=self.dtype,
            device=self.device,
        )  # Target condition

        y = torch.ones(
            (batch_size,) + (input_channels,) + self.spatial_resolution,
            dtype=self.dtype,
            device=self.device,
        )  # Initial condition

        if time_cond:
            time = torch.ones((batch_size,), dtype=self.dtype, device=self.device)
        else:
            time = None

        return self.denoiser(
            x=x,
            y=y,
            sigma=torch.ones((batch_size,), dtype=self.dtype, device=self.device),
            time=time,
        )

    def loss_fn(self, batch: dict, mutables: Optional[dict] = None):
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

        y = batch["initial_cond"]
        x = batch["target_cond"]
        time = batch["lead_time"] if self.time_cond else None

        batch_size = len(x)

        x_squared = torch.square(x)

        sigma = self.noise_sampling(shape=(batch_size,))

        weights = self.noise_weighting(sigma)
        if weights.ndim != x.ndim:
            weights = weights.view(-1, *([1] * (x.ndim - 1)))

        noise = torch.randn(x.shape, dtype=self.dtype, device=self.device)

        if sigma.ndim != x.ndim:
            noised = x + noise * sigma.view(-1, *([1] * (x.ndim - 1)))
        else:
            noised = x + noise * sigma

        if time is not None:
            denoised = self.denoiser.forward(x=noised, y=y, sigma=sigma, time=time)
        else:
            denoised = self.denoiser.forward(x=noised, y=y, sigma=sigma)

        denoised_squared = torch.square(denoised)

        rel_norm = torch.mean(torch.square(x) / torch.mean(torch.square(x_squared)))
        loss = torch.mean(weights * torch.square(denoised - x))
        loss += (
            self.consistent_weight
            * rel_norm
            * torch.mean(weights * torch.square(denoised_squared - x_squared))
        )

        # Additional metrics can be stored here
        metrics = {"loss": loss.item()}

        return loss, metrics

    def eval_fn(self, batch: dict) -> dict:
        """Compute denoising metrics on an eval batch.

        Randomly selects members of the batch and noise them to a number of fixed
        levels. Each level is aggregated in terms of the average L2 error.

        Args:
          variables: Variables for the denoising module.
          batch: A batch of evaluation data expected to contain an `x` field with a
            shape of `(batch, *spatial_dims, channels)`, representing the unnoised
            samples. Optionally, it may also contain a `cond` field, which is a
            dictionary of conditional inputs.

        Returns:
          A dictionary of denoising-based evaluation metrics.
        """

        initial_cond = batch["initial_cond"]
        target_cond = batch["target_cond"]
        time = batch["lead_time"] if self.time_cond else None

        rand_idx_set = torch.randint(
            0,
            initial_cond.shape[0],
            (self.num_eval_noise_levels, self.num_eval_cases_per_lvl),
            device=self.device,
        )

        y = initial_cond[rand_idx_set]
        x = target_cond[rand_idx_set]

        if time is not None:
            time_inputs = time[rand_idx_set]

        sigma = torch.exp(
            torch.linspace(
                np.log(self.min_eval_noise_lvl),
                np.log(self.max_eval_noise_lvl),
                self.num_eval_noise_levels,
                dtype=self.dtype,
                device=self.device,
            )
        )

        noise = torch.randn(x.shape, device=self.device, dtype=self.dtype)

        if sigma.ndim != x.ndim:
            noised = x + noise * sigma.view(-1, *([1] * (x.ndim - 1)))
        else:
            noised = x + noise * sigma

        denoise_fn = self.inference_fn(
            denoiser=self.denoiser,
            task=self.task,
            lead_time=False if time is None else True,
        )

        if time is not None:
            denoised = torch.stack(
                [
                    denoise_fn(
                        x=noised[i],
                        y=y[i],
                        sigma=sigma[i].unsqueeze(0),
                        time=time_inputs[i],
                    )
                    for i in range(self.num_eval_noise_levels)
                ]
            )
        else:
            denoised = torch.stack(
                [
                    denoise_fn(x=noised[i], y=y[i], sigma=sigma[i])
                    for i in range(self.num_eval_noise_levels)
                ]
            )

        ema_losses = torch.mean(
            torch.square(denoised - x), dim=[i for i in range(1, x.ndim)]
        )
        eval_losses = {
            f"denoise_lvl{i}": loss.item() for i, loss in enumerate(ema_losses)
        }
        return eval_losses

    @staticmethod
    def inference_fn(
        denoiser: nn.Module, task: str = "solver", lead_time: bool = False
    ) -> Tensor:
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

        if task == "superresolver":

            def _denoise(
                x: Tensor,
                sigma: float | Tensor,
                cond: Mapping[str, Tensor] | None = None,
            ) -> Tensor:

                if not torch.is_tensor(sigma):
                    sigma = sigma * torch.ones((x.shape[0],))

                return denoiser.forward(x=x, sigma=sigma)

        elif task == "solver" and lead_time == False:

            def _denoise(
                x: Tensor,
                sigma: float | Tensor,
                y: Tensor,
                cond: Mapping[str, Tensor] | None = None,
            ) -> Tensor:

                if not torch.is_tensor(sigma):
                    sigma = sigma * torch.ones((x.shape[0],))

                return denoiser.forward(x=x, sigma=sigma, y=y)

        elif task == "solver" and lead_time == True:

            def _denoise(
                x: Tensor,
                sigma: float | Tensor,
                y: Tensor,
                time: float | Tensor,
                cond: Mapping[str, Tensor] | None = None,
            ) -> Tensor:

                if not torch.is_tensor(sigma):
                    sigma = sigma * torch.ones((x.shape[0],))

                if not torch.is_tensor(time):
                    time = time * torch.ones((x.shape[0],))

                return denoiser.forward(x=x, sigma=sigma, y=y, time=time)

        else:
            raise ValueError(
                "model can either be used as a 'superresolver' or a 'solver'"
            )

        return _denoise
