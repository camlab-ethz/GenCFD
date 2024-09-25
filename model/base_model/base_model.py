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

from abc import ABC, abstractmethod
from typing import Callable, Mapping, Any, Union, Tuple

import torch
import torch.nn as nn
import numpy as np

Tensor = torch.Tensor
TensorDict = Mapping[str, Tensor]
BatchType = Mapping[str, Union[np.ndarray, Tensor]]
ModelVariable = Union[dict, tuple[dict, ...], Mapping[str, dict]]
PyTree = Any
LossAndAux = tuple[Tensor, tuple[TensorDict, PyTree]]


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
