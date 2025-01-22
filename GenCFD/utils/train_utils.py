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

"""Utility functions for the template."""

import collections
from collections.abc import Callable, Mapping, Sequence
import functools
from typing import Any, Union
from functools import wraps

import torch
import numpy as np
import torch.optim as optim
from tensorboard.backend.event_processing import event_accumulator
from torchmetrics import Metric

Scalar = Any
Tensor = torch.Tensor


class StdMetric(Metric):
    """Computes the standard deviation of a stream of values."""

    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "sum_of_squares", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, values: Union[Tensor, float]):
        """Accumulate values for the standard deviation computation."""

        # Ensure the input is a tensor
        if isinstance(values, float):
            values = torch.as_tensor(values)

        self.total += values.sum()
        self.sum_of_squares += (values**2).sum()
        self.count += values.numel()

    def compute(self):
        """Compute the standard deviation."""
        mean = self.total / self.count
        variance = (self.sum_of_squares / self.count) - mean**2
        # In theory the variance is non negative, but we can ensure this
        variance = torch.clamp(variance, min=0.0)
        return torch.sqrt(variance)


def compute_memory(func):
    """Decorator to track GPU memory usage if `track_memory` is enabled."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.device.type == "cuda" and self.track_memory:
            # Reset peak memory
            torch.cuda.reset_peak_memory_stats(self.device)

        metrics = func(self, *args, **kwargs)

        if self.device.type == "cuda" and self.track_memory:
            # Compute the peak memory
            peak_mem = torch.cuda.max_memory_allocated(self.device) / (
                1024**3
            )  # Convert to GB
            # Inject memory info into metrics
            if isinstance(metrics, dict):  # Assuming metrics is a dictionary
                metrics["mem"] = peak_mem

        return metrics

    return wrapper


def primary_process_only(cls: type[Any]) -> type[Any]:
    """Class decorator that modifies all methods to run on primary host only."""

    def wrap_method(method: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if torch.distributed.get_rank() == 0:
                return method(self, *args, **kwargs)
            else:
                return None

        return wrapper

    for attr_name, attr_value in cls.__dict__.items():
        if callable(attr_value) and not attr_name.startswith("__"):
            setattr(cls, attr_name, wrap_method(attr_value))

    return cls


def load_scalars_from_tfevents(
    logdir: str,
) -> Mapping[int, Mapping[str, Scalar]]:
    """Loads scalar summaries from events in a logdir."""
    event_acc = event_accumulator.EventAccumulator(logdir)
    event_acc.Reload()

    data = collections.defaultdict(dict)

    for tag in event_acc.Tags()["scalars"]:
        for scalar_event in event_acc.Scalars(tag):
            data[scalar_event.step][tag] = scalar_event.value

    return data


def is_scalar(value: Any) -> bool:
    """Checks if a given value is a scalar."""
    if isinstance(value, (int, float, np.number)):
        return True
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.ndim == 0 or value.numel() <= 1
    return False


def opt_chain(
    transformations: Sequence[optim.Optimizer],
) -> optim.Optimizer:
    """Wraps `optax.chain` to allow keyword arguments (for gin config)."""
    if len(transformations) == 1:
        return transformations[0]
    else:
        raise NotImplementedError(
            "PyTorch does not support chaining optimizers. Use custom optimizer Logic."
        )


def create_slice(
    start: int | None = None, end: int | None = None, step: int | None = None
) -> slice:
    """Wraps the python `slice` to allow keyword arguments (for gin config)."""
    return slice(start, end, step)
