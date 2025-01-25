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

"""Function that runs training."""

import os
from typing import Any, Sequence, Optional
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from GenCFD.utils import callbacks as cb
from GenCFD.train import trainers
from GenCFD.utils import train_utils


def run(
    *,
    train_dataloader: DataLoader,
    trainer: trainers.BaseTrainer,
    workdir: str,
    # DDP configs
    world_size: int,
    local_rank: int,
    # training configs
    total_train_steps: int,
    metric_aggregation_steps: int = 50,
    # evaluation configs
    eval_dataloader: Optional[DataLoader] = None,
    eval_every_steps: int = 100,
    num_batches_per_eval: int = 10,
    run_sanity_eval_batch: bool = True,
    # other configs
    metric_writer: Optional[SummaryWriter] = None,
    callbacks: Sequence[cb.Callback] = (),
) -> None:
    """Runs trainer for a training task.

    This function runs a trainer in batches of "metric aggregation" steps, where
    the step-wise metrics obtained within the same batch are aggregated
    (i.e. by computing the average and/or std based on the metric defined in
    the trainer class). The aggregated metrics are then automatically saved to a
    tensorflow event file in `workdir`. Evaluation runs periodically, i.e. once
    every `eval_every_steps` steps, if an eval dataloader is provided.

    Args:
      train_dataloader: A dataloader emitting training data in batches.
      trainer: A trainer object hosting the train and eval logic.
      workdir: The working directory where results (e.g. train & eval metrics) and
        progress (e.g. checkpoints) are saved.
      world_size: describes if model is in ddp mode trained (world_size > 1)
      total_train_steps: Total number of training steps to run.
      metric_aggregation_steps: The trainer runs this number of steps at a time,
        after which training metrics are aggregated and logged.
      eval_dataloader: An evaluation dataloader (optional). If set to `None`, no
        evaluation will run.
      eval_every_steps: The period, in number of train steps, at which evaluation
        runs. Must be an integer multiple of `metric_aggregation_steps`.
      num_batches_per_eval: The number of batches to step through every time
        evaluation is run (resulting metrics are aggregated).
      run_sanity_eval_batch: Whether to step through sanity check eval batch
        before training starts. This helps expose runtime issues early, without
        having to wait until evaluation is first triggered (i.e. after
        `eval_every_steps`).
      metric_writer: A metric writer that writes scalar metrics to disc. It is
        also accessible to callbacks for custom writing in other formats.
      callbacks: A sequence of self-contained programs executing non-essential
        logic (e.g. checkpoint saving, logging, timing, profiling etc.).
    """
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    train_iter = iter(train_dataloader)
    eval_iter = None
    run_evaluation = eval_dataloader is not None
    if run_evaluation:
        if eval_every_steps % metric_aggregation_steps != 0:
            raise ValueError(
                f"`eval_every_steps` ({eval_every_steps}) "
                f"must be an integer multiple of "
                f"`metric_aggregation_steps` ({metric_aggregation_steps})"
            )

        eval_iter = iter(eval_dataloader)
        if run_sanity_eval_batch and not trainer.is_compiled:
            trainer.eval(eval_iter, num_steps=1)

    for callback in callbacks:
        callback.metric_writer = metric_writer if (local_rank in [0, -1] and metric_writer) else None
        callback.on_train_begin(trainer)

    cur_step = trainer.train_state.int_step

    # setup for reinitializing iterator for training and evaluation
    if run_evaluation:
        epoch_eval = 1
        step_diff_eval = 1 if run_sanity_eval_batch else 0
        eval_steps_per_epoch = (
            len(eval_dataloader) // num_batches_per_eval * eval_every_steps
        )
        epochs_eval_steps = epoch_eval * eval_steps_per_epoch - step_diff_eval

    epoch_train = 1
    step_diff_train = 0
    epochs_train_steps = epoch_train * len(train_dataloader) - step_diff_train

    # Barrier before training
    if world_size > 1:
        dist.barrier(device_ids=[local_rank])

    while cur_step < total_train_steps:
        for callback in callbacks:
            callback.on_train_batches_begin(trainer)

        num_steps = min(total_train_steps - cur_step, metric_aggregation_steps)

        # evaluate if training dataset reinitialization is necessary
        if cur_step + num_steps > epochs_train_steps:
            epoch_train += 1  # increase epoch for training dataset

            if world_size > 1:
                # Reset for random shuffling
                train_dataloader.sampler.set_epoch(epoch_train)

            train_iter = iter(train_dataloader)
            step_diff_train += epochs_train_steps - cur_step
            epochs_train_steps = epoch_train * len(train_dataloader) - step_diff_train

        train_metrics = trainer.train(train_iter, num_steps).compute()
        cur_step += num_steps

        if local_rank in [0, -1] and metric_writer:
            metric_writer.add_scalars("train", train_metrics, cur_step)

        # At train/eval batch end, callbacks are called in reverse order so that
        # they are last-in-first-out, loosely resembling nested python contexts.
        for callback in reversed(callbacks):
            callback.on_train_batches_end(trainer, train_metrics)

        if run_evaluation:
            if cur_step == total_train_steps or cur_step % eval_every_steps == 0:
                for callback in callbacks:
                    callback.on_eval_batches_begin(trainer)

                assert eval_iter is not None

                # evaluate if evaluation iterator needs to be reinitialized
                if cur_step + num_batches_per_eval > epochs_eval_steps:
                    epoch_eval += 1  # increase epoch for evaluation dataset

                    if world_size > 1:
                        # Reset for random shuffling
                        eval_dataloader.sampler.set_epoch(epoch_eval)

                    eval_iter = iter(eval_dataloader)
                    step_diff_eval += epochs_eval_steps - cur_step
                    epochs_eval_steps = (
                        epoch_eval * eval_steps_per_epoch - step_diff_eval
                    )

                eval_metrics = trainer.eval(eval_iter, num_batches_per_eval).compute()
                eval_metrics_to_log = {
                    k: v for k, v in eval_metrics.items() if train_utils.is_scalar(v)
                }

                if local_rank in [0, -1] and metric_writer:
                    metric_writer.add_scalars("eval", eval_metrics_to_log, cur_step)

                for callback in reversed(callbacks):
                    callback.on_eval_batches_end(trainer, eval_metrics)

    for callback in reversed(callbacks):
        callback.on_train_end(trainer)

    if local_rank in [0, -1] and metric_writer:
        metric_writer.flush()
