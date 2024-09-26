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

"""Trainer classes for use in gradient descent mini-batch training."""

import abc
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Generic, TypeVar

# from clu import metrics as clu_metrics
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from model.base_model import base_model
from train import train_states
from torchmetrics import MetricCollection

Tensor = torch.Tensor 
BatchType = Mapping[str, Tensor]
Metrics = MetricCollection
# PyTree = Any

M = TypeVar("M")  # Model
S = TypeVar("S", bound=train_states.BasicTrainState)  # Train state


class BaseTrainer(Generic[M, S], metaclass=abc.ABCMeta):
  """Abstract base trainer for gradient descent mini-batch training."""

  def __init__(self, model: M, device: torch.device):
    self.model = model.to(device)
    self.device = device
    self.train_state = self.initialize_train_state()
    self._compiled_train_step = self.train_step
    self._compiled_eval_step = self.eval_step

  @property
  @abc.abstractmethod
  def train_step(self) -> Callable[[S, BatchType], tuple[S, Metrics]]:
      """Returns the train step function."""
      raise NotImplementedError

  @property
  @abc.abstractmethod
  def eval_step(self) -> Callable[[S, BatchType], Metrics]:
      """Returns the evaluation step function."""
      raise NotImplementedError

  @abc.abstractmethod
  def initialize_train_state(self) -> S:
      """Instantiate the initial train state."""
      raise NotImplementedError

  def train(self, batch_iter: Iterator[BatchType], num_steps: int) -> Metrics:
      """Runs training for a specified number of steps."""
      train_metrics = self.TrainMetrics()
      self.model.train()

      for step in range(num_steps):
          batch = next(batch_iter)
          batch = {k: v.to(self.device) for k, v in batch.items()}
          self.train_state, metrics_update = self._compiled_train_step(batch) # self.train_state as first entry
          train_metrics.update(metrics_update.train_loss)

      return train_metrics

  def eval(self, batch_iter: Iterator[BatchType], num_steps: int) -> Metrics:
      """Runs evaluation for a specified number of steps."""
      eval_metrics = self.EvalMetrics()
      self.model.eval()

      with torch.no_grad():
          for _ in range(num_steps):
              batch = next(batch_iter)
              batch = {k: v.to(self.device) for k, v in batch.items()}
              metrics_update = self._compiled_eval_step(batch) # self.train_state as first entry
              eval_metrics.update(metrics_update.eval_accuracy)

      return eval_metrics
  

class BasicTrainer(BaseTrainer[M, S]):
  """Basic Trainer implementing the training/evaluation steps."""

  class TrainMetrics(Metrics):
      """Training metrics based on the model outputs."""
      # Example usage:
      # train_loss = torchmetrics.MeanSquaredError()
      # train_acc = torchmetrics.Accuracy()

  class EvalMetrics(Metrics):
      """Evaluation metrics based on model outputs."""
      # Example usage:
      # eval_loss = torchmetrics.MeanSquaredError()
      # eval_acc = torchmetrics.Accuracy()

  def __init__(self, model: nn.Module, optimizer: optim.Optimizer, device: torch.device):
      super().__init__(model, device)
      self.optimizer = optimizer

  @property
  def train_step(self) -> Callable[[S, BatchType], tuple[S, Metrics]]:
      def _train_step(train_state: S, batch: BatchType) -> tuple[S, Metrics]:
          """Performs gradient step and computes training metrics."""
          self.model.train()
          output = self.model(batch)
          loss, metrics = self.model.loss_fn(output, batch)

          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

          # Update the train state with the new model parameters
          new_state = train_state.update(
              step=train_state.step + 1,
              params=self.model.state_dict()
          )

          # Update metrics
          metrics_update = self.TrainMetrics(**metrics)
          return new_state, metrics_update

      return _train_step

  @property
  def eval_step(self) -> Callable[[S, BatchType], Metrics]:
      def _eval_step(train_state: S, batch: BatchType) -> Metrics:
          """Evaluates the model and computes evaluation metrics."""
          self.model.eval()
          with torch.no_grad():
              output = self.model(batch)
              loss, metrics = self.model.eval_fn(output, batch)

          metrics_update = self.EvalMetrics(**metrics)
          return metrics_update

      return _eval_step

  def initialize_train_state(self) -> S:
      """Initializes the training state, including optimizer and parameters."""
      return train_states.BasicTrainState.create(
          params=self.model.state_dict(),
          opt_state=self.optimizer.state_dict()
      )


class BasicDistributedTrainer(BasicTrainer[M, S]):
  """Distributed Trainer for DDP (DistributedDataParallel) training."""

  def __init__(self, model: nn.Module, optimizer: optim.Optimizer, device: torch.device):
      super().__init__(model, optimizer, device)
      self.model = DDP(self.model, device_ids=[device])

  def train_step(self, train_state: S, batch: BatchType) -> tuple[S, Metrics]:
      return super().train_step(train_state, batch)

  def eval_step(self, train_state: S, batch: BatchType) -> Metrics:
      return super().eval_step(train_state, batch)