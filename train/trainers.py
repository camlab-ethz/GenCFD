# Copyright 2024 The swirl_dynamics Authors and Modifications made 
# by the CAM Lab at ETH Zurich.
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
import functools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from torchmetrics import MetricCollection, MeanMetric

from model.base_model import base_model
from train import train_states
import diffusion as dfn_lib

Tensor = torch.Tensor 
BatchType = Mapping[str, Tensor]
Metrics = MetricCollection
# PyTree = Any

M = TypeVar("M")  # Model
S = TypeVar("S", bound=train_states.BasicTrainState)  # Train state
D = TypeVar("D", bound=dfn_lib.DenoisingModel)
SD = TypeVar("SD", bound=train_states.DenoisingModelTrainState)


class BaseTrainer(Generic[M, S], metaclass=abc.ABCMeta):
    """Abstract base trainer for gradient descent mini-batch training."""

    def __init__(self, 
                 model: M,
                 device: torch.device):
        # self.model = model.to(device)
        self.model = model
        self.device = device
        self.train_state = self.initialize_train_state()

    @property
    @abc.abstractmethod
    def train_step(self) -> Metrics:
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
        self.model.denoiser.train() # adapt this part!

        for step in range(num_steps):
            batch = next(batch_iter)
            # batch = {k: v.to(self.device) for k, v in batch.items()}
            metrics_update = self.train_step(batch)
            # TODO: Write an if statement that keeps track of a mean lets say after every 10th or 50th iteration!!!
            train_loss_value = metrics_update["train_loss"].compute()
            train_metrics.update(train_loss_value)
            # print(f"mean_loss: {train_loss_value}")

        return train_metrics

    def eval(self, batch_iter: Iterator[BatchType], num_steps: int) -> Metrics:
        """Runs evaluation for a specified number of steps."""
        eval_metrics = self.EvalMetrics()
        self.model.eval()

        with torch.no_grad():
            for _ in range(num_steps):
                batch = next(batch_iter)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                metrics_update = self.eval_step(batch) # self.train_state as first entry
                eval_metrics.update(metrics_update.eval_accuracy)

        return eval_metrics
  

class BasicTrainer(BaseTrainer[M, S]):
    """Basic Trainer implementing the training/evaluation steps."""

    class TrainMetrics(Metrics):
        """Training metrics based on the model outputs."""
        # Example usage:
        # train_loss = MeanMetric()
        # train_acc = torchmetrics.Accuracy()
        def __init__(self):
            metrics = {
                # 'train_loss': MeanMetric(),
                #'train_acc': torchmetrics.Accuracy()
            }
            super().__init__(metrics)

    class EvalMetrics(Metrics):
        """Evaluation metrics based on model outputs."""
        # Example usage:
        # eval_loss = torchmetrics.MeanSquaredError()
        # eval_acc = torchmetrics.Accuracy()

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, device: torch.device):
        super().__init__(model, device)
        self.optimizer = optimizer


    def train_step(self, batch: BatchType) -> Metrics:
        
        self.model.train()
        output = self.model(batch)
        loss, metrics = self.model.loss_fn(output, batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_train_state()

        train_metrics = self.TrainMetrics()
        train_metrics.update(torch.tensor(metrics["train_loss"]))

        return train_metrics


    def eval_step(self, batch: BatchType) -> Callable[[S, BatchType], Metrics]:
        # TODO: NOT WORKING RN, IMPLEMENT THIS IN THE SAME WAY AS TRAIN_STEP
        self.model.eval()
        with torch.no_grad():
            output = self.model(batch)
            loss, metrics = self.model.eval_fn(output, batch)
        metrics_update = self.EvalMetrics(**metrics)
        return metrics_update


    def initialize_train_state(self) -> S:
        """Initializes the training state, including optimizer and parameters."""
        return train_states.BasicTrainState(
            model=self.model,
            optimizer=self.optimizer,
            params=self.model.state_dict(),
            opt_state=self.optimizer.state_dict()
        )
    
    def update_train_state(self) -> S:
        """Update the training state, including optimizer and parameters."""
        next_step = self.train_state.step + 1
        if isinstance(next_step, Tensor):
            next_step = next_step.item()

        return self.train_state.replace(
            step=next_step,
            opt_state=self.optimizer.state_dict(),
            params=self.model.state_dict()
        )


class BasicDistributedTrainer(BasicTrainer[M, S]):
    """Distributed Trainer for DDP (DistributedDataParallel) training."""

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, device: torch.device):
        super().__init__(model, optimizer, device)
        self.model = DDP(self.model, device_ids=[device])

    def train_step(self, batch: BatchType) -> Metrics:
        return super().train_step(batch)

    def eval_step(self, batch: BatchType) -> Metrics:
        return super().eval_step(batch)


class DenoisingTrainer(BasicTrainer[M, SD]):
    def __init__(
            self, 
            model: nn.Module, 
            optimizer: optim.Optimizer,
            device: torch.device,
            ema_decay: float = 0.999):
      
      self.optimizer = optimizer
      self.ema_decay = ema_decay 

      super().__init__(model=model, optimizer=optimizer, device=device)

    class TrainMetrics(Metrics):
        """Train metrics including mean and std of loss"""
        def __init__(self):
            train_metrics = {
                "train_loss": MeanMetric(),
                "train_loss_std": MeanMetric() # TODO: Change to STD!
            }
            super().__init__(metrics=train_metrics)
    

    class EvalMetrics(Metrics):
        """Evaluation metrics based on the model output, using noise level"""
        def __init__(self, num_eval_noise_levels: int = 10):
            eval_metrics = {
                f"eval_denoise_lvl{i}": MeanMetric() 
                for i in range(num_eval_noise_levels) 
            }
            super().__init__(metrics=eval_metrics)
    

    def initialize_train_state(self) -> SD:
        """Initializes the train state with EMA and model params"""
        return train_states.DenoisingModelTrainState(
            model=self.model.denoiser,
            optimizer=self.optimizer,
            params=self.model.denoiser.state_dict(),
            opt_state=self.optimizer.state_dict(),
            step=0,
            ema_decay=self.ema_decay
        )
    
    def train_step(self, batch: BatchType) -> Metrics:
        
        self.model.denoiser.train()
        #TODO: Changed to device!
        loss, (metrics, mem) = self.model.loss_fn(batch[0].to(device=self.device))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_train_state()

        train_metrics = self.TrainMetrics()

        # train_metrics.update(torch.tensor(metrics["loss"]))
        train_metrics.update(metrics["loss"])

        # print(f"LOSS: {loss}")

        return train_metrics
    

    def update_train_state(self) -> SD:
        """Update the training state, including optimizer and parameters."""
        next_step = self.train_state.step + 1
        if isinstance(next_step, Tensor):
            next_step = next_step.item()

        # update ema model
        self.train_state.ema_model.update_parameters(self.model.denoiser)
        ema_params = self.train_state.ema_parameters

        return self.train_state.replace(
            step=next_step,
            opt_state=self.optimizer.state_dict(),
            params=self.model.denoiser.state_dict(),
            ema=ema_params,
        )
    
    @staticmethod
    def inference_fn_from_state_dict(
        state: SD,
        *args,
        use_ema: bool = True,
        **kwargs
    ):
        if use_ema:
            if state.ema_model:
                variables = state.ema_parameters
            else:
                raise ValueError("EMA model is None or not initialized")
            
        else:
            variables = state.model.state_dict()

        return dfn_lib.DenoisingModel.inference_fn(variables, *args, **kwargs)

    
