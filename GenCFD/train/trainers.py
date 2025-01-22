# Copyright 2024 The swirl_dynamics Authors.
# Modifications made by the CAM Lab at ETH Zurich
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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from torchmetrics import MetricCollection, MeanMetric
from GenCFD.utils.train_utils import StdMetric, compute_memory

from GenCFD.train import train_states
import GenCFD.diffusion as dfn_lib

Tensor = torch.Tensor
BatchType = Mapping[str, Tensor]
Metrics = MetricCollection

M = TypeVar("M")  # Model
S = TypeVar("S", bound=train_states.BasicTrainState)
D = TypeVar("D", bound=dfn_lib.DenoisingModel)
SD = TypeVar("SD", bound=train_states.DenoisingModelTrainState)


class BaseTrainer(Generic[M, S], metaclass=abc.ABCMeta):
    """Abstract base trainer for gradient descent mini-batch training."""

    def __init__(
        self,
        model: M,
        device: torch.device = None,
        track_memory: bool = False,
        world_size: int = 1,
        local_rank: int = -1,
    ):
        self.model = model
        self.device = device
        self.train_state = self.initialize_train_state()
        self.track_memory = track_memory
        self.world_size = world_size
        self.local_rank = local_rank

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

        train_metrics = self.TrainMetrics(
            device=self.device,
            track_memory=self.track_memory,
            world_size=self.world_size,
        )
        self.model.denoiser.train()

        for step in range(num_steps):
            batch = next(batch_iter)
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            metrics_update = self.train_step(batch)

            train_metrics["loss"].update(metrics_update["loss"])
            train_metrics["loss_std"].update(metrics_update["loss"])

            if self.track_memory and "mem" in metrics_update:
                train_metrics["mem"].update(metrics_update["mem"])

        if self.track_memory and self.device.type != "cuda":
            print(f"Warning: Memory tracking is skipped. CUDA device is not available.")

        if self.world_size > 1:
            # Barrier / Synchronization before training aggregation
            dist.barrier(device_ids=[self.local_rank])

        return train_metrics

    def eval(self, batch_iter: Iterator[BatchType], num_steps: int) -> Metrics:
        """Runs evaluation for a specified number of steps."""
        eval_metrics = self.EvalMetrics(
            self.device, self.model.num_eval_noise_levels, self.world_size
        )
        self.model.denoiser.eval()

        with torch.no_grad():
            for _ in range(num_steps):
                batch = next(batch_iter)
                batch = {
                    k: v.to(self.device, non_blocking=True) for k, v in batch.items()
                }
                update_metrics = self.eval_step(
                    batch
                )  # self.train_state as first entry
                for key, value in update_metrics.items():
                    eval_metrics[key].update(value)

        if self.world_size > 1:
            # Barrier / Synchronization before evaluation aggregation
            dist.barrier(device_ids=[self.local_rank])

        return eval_metrics


class BasicTrainer(BaseTrainer[M, S]):
    """Basic Trainer implementing the training/evaluation steps."""

    class TrainMetrics(Metrics):
        """Training metrics based on the model outputs."""

        # Example usage:
        # train_loss = MeanMetric()
        # train_acc = torchmetrics.Accuracy()
        # memory tracer if set to True
        def __init__(self, device, track_memory: bool = False):
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

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device = None,
        track_memory: bool = False,
        world_size: int = 1,
        local_rank: int = -1,
    ):
        super().__init__(
            model=model,
            device=device,
            track_memory=track_memory,
            world_size=world_size,
            local_rank=local_rank,
        )
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
        train_metrics.update(torch.tensor(metrics["loss"]))

        return train_metrics

    def eval_step(self, batch: BatchType) -> Callable[[S, BatchType], Metrics]:
        with torch.no_grad():
            metrics = self.model.eval_fn(batch)

        eval_metrics = self.EvalMetrics(
            self.device, self.model.num_eval_noise_levels, self.world_size
        )
        for key, value in metrics.items():
            eval_metrics[key](value)

        return eval_metrics.compute()

    def initialize_train_state(self) -> S:
        """Initializes the training state, including optimizer and parameters."""
        return train_states.BasicTrainState(
            model=self.model,
            optimizer=self.optimizer,
            params=self.model.state_dict(),
            opt_state=self.optimizer.state_dict(),
        )

    def update_train_state(self) -> S:
        """Update the training state, including optimizer and parameters."""
        next_step = self.train_state.step + 1
        if isinstance(next_step, Tensor):
            next_step = next_step.item()

        return self.train_state.replace(
            step=next_step,
            opt_state=self.optimizer.state_dict(),
            params=self.model.state_dict(),
        )


class BasicDistributedTrainer(BasicTrainer[M, S]):
    """Distributed Trainer for DDP (DistributedDataParallel) training."""

    def __init__(
        self, model: nn.Module, optimizer: optim.Optimizer, device: torch.device
    ):
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
        ema_decay: float = 0.999,
        store_ema: bool = False,
        track_memory: bool = False,
        use_mixed_precision: bool = False,
        is_compiled: bool = False,
        world_size: int = 1,
        local_rank: int = -1,
    ):

        self.optimizer = optimizer
        self.ema_decay = ema_decay
        self.store_ema = store_ema
        self.track_memory = track_memory
        # Mixed precision training with Grad scaler to avoid overflow and underflow during backprop.
        self.compute_dtype = torch.float16 if use_mixed_precision else torch.float32
        self.use_mixed_precision = use_mixed_precision
        self.scaler = torch.amp.GradScaler(device.type) if use_mixed_precision else None
        # Store status if the model is compiled and / or parallellized
        self.is_compiled = is_compiled
        self.is_parallelized = True if world_size > 1 else False

        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            track_memory=track_memory,
            world_size=world_size,
            local_rank=local_rank,
        )

    class TrainMetrics(Metrics):
        """Train metrics including mean and std of loss and if required
        computes the mean of the memory profiler."""

        def __init__(self, device, track_memory: bool = False, world_size: int = 1):
            train_metrics = {
                "loss": MeanMetric(
                    sync_on_compute=True if world_size > 1 else False
                ).to(device),
                "loss_std": StdMetric().to(
                    device
                ),  # uses already reduction clauses thus no sync
            }
            if track_memory:
                train_metrics["mem"] = MeanMetric(
                    sync_on_compute=True if world_size > 1 else False
                ).to(device)

            super().__init__(metrics=train_metrics)

    class EvalMetrics(Metrics):
        """Evaluation metrics based on the model output, using noise level"""

        def __init__(self, device, num_eval_noise_levels: int, world_size: int = 1):
            eval_metrics = {
                f"denoise_lvl{i}": MeanMetric(
                    sync_on_compute=True if world_size > 1 else False
                ).to(device)
                for i in range(num_eval_noise_levels)
            }
            super().__init__(metrics=eval_metrics)

    def initialize_train_state(self) -> SD:
        """Initializes the train state with EMA and model params

        Those states are tracked at every iteration step
        """
        return train_states.DenoisingModelTrainState(
            # Further parameters can be added here to track
            model=self.model.denoiser if self.store_ema else None,
            step=0,
            ema_decay=self.ema_decay,
            store_ema=self.store_ema,
        )

    @compute_memory
    def train_step(self, batch: BatchType) -> Metrics:

        with torch.amp.autocast(device_type=self.device.type, dtype=self.compute_dtype):
            loss, metrics = self.model.loss_fn(batch)

        self.optimizer.zero_grad(set_to_none=True)
        if self.use_mixed_precision:
            loss = loss.float()
            self.scaler.scale(loss).backward(retain_graph=False)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward(retain_graph=False)
            self.optimizer.step()

        self.update_train_state()

        return metrics

    def update_train_state(self) -> SD:
        """Update the training state, including optimizer and parameters."""
        next_step = self.train_state.step + 1
        if isinstance(next_step, Tensor):
            next_step = next_step.item()

        # update ema model
        if self.store_ema:
            self.train_state.ema_model.update_parameters(self.model.denoiser)
            ema_params = self.train_state.ema_parameters

        # Further states can be replaced at every training step
        return self.train_state.replace(
            step=next_step, ema=ema_params if self.store_ema else None
        )

    @staticmethod
    def inference_fn_from_state_dict(
        state: SD,
        denoiser: nn.Module,
        *args,
        use_ema: bool = False,
        task: str = "solver",
        lead_time: bool = False,
        **kwargs,
    ):
        denoiser.eval()
        if use_ema:
            if state.ema_model:
                denoiser.load_state_dict(state.ema_parameters)

            else:
                raise ValueError("EMA model is None or not initialized")

        return dfn_lib.DenoisingModel.inference_fn(
            denoiser, task, lead_time, *args, **kwargs
        )
