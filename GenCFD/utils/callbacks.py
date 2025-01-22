# Copyright 2024 The swirl_dynamics Authors.
# Modifications made by The CAM Lab at ETH Zurich.
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

"""Training callback library."""

from collections.abc import Mapping, Sequence
import os
import time
from typing import Any, Optional

import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm

from GenCFD.train import trainers

Tensor = torch.Tensor
ComputedMetrics = Mapping[str, Tensor | Mapping[str, Tensor]]
Trainer = trainers.BaseTrainer


class Callback:
    """Abstract base class for callbacks.

    Callbacks are self-contained programs containing some common, reusable logic
    that is non-essential (such as saving model checkpoints, reporting progress,
    profiling, the absence of which would not "break" training) to model training.
    The instance methods of these objects are hooks that get executed at various
    phases of training (i.e. fixed positions inside `train.run` function).

    The execution (in `train.run`) observes the following flow::

      callbacks.on_train_begin()
      while training:
        callbacks.on_train_batches_begin()
        run_train_steps()
        callbacks.on_train_batches_end()
        if should_run_evaluation:
          callbacks.on_eval_batches_begin()
          run_eval_steps()
          callbacks.on_eval_batches_end()
      callbacks.on_train_end()

    The hooks may read and/or overwrite the trainer state and/or train/eval
    metrics, and have access to a metric_writer that writes desired info/variables
    to the working directory in tensorflow event format.

    When multiple (i.e. a list of) callbacks are used, the
    `on_{train/eval}_batches_end` methods are called in reverse order (so that
    together with `on_{train/eval}_batches_begin` calls they resemble
    the `__exit__` and `__enter__` methods of python contexts).
    """

    def __init__(self, log_dir: Optional[str] = None):
        """Initializes the callback with an optional log directory for metrics."""
        self._metric_writer = None
        if log_dir:
            self._metric_writer = SummaryWriter(log_dir=log_dir)

    @property
    def metric_writer(self) -> SummaryWriter:
        """Property for the metric writer."""
        assert hasattr(self, "_metric_writer")
        return self._metric_writer

    @metric_writer.setter
    def metric_writer(self, writer: SummaryWriter) -> None:
        self._metric_writer = writer

    def on_train_begin(self, trainer: Trainer) -> None:
        """Called before the training loop starts."""
        # if self.metric_writer:
        #   self.metric_writer.add_text("Train", "Training started.")

    def on_train_batches_begin(self, trainer: Trainer) -> None:
        """Called before a training segment begins."""

    def on_train_batches_end(
        self, trainer: Trainer, train_metrics: ComputedMetrics
    ) -> None:
        """Called after a training segment ends."""
        # if self.metric_writer:
        #   for metric_name, metric_value in train_metrics.items():
        #     self.metric_writer.add_scalar(f"Train/{metric_name}", metric_value, trainer.train_state.step)

    def on_eval_batches_begin(self, trainer: Trainer) -> None:
        """Called before an evaluation segment begins."""

    def on_eval_batches_end(
        self, trainer: Trainer, eval_metrics: ComputedMetrics
    ) -> None:
        """Called after an evaluation segment ends."""
        # if self.metric_writer:
        #   for metric_name, metric_value in eval_metrics.items():
        #     self.metric_writer.add_scalar(f"Eval/{metric_name}", metric_value, trainer.train_state.step)

    def on_train_end(self, trainer: Trainer) -> None:
        """Called when training ends."""
        # if self.metric_writer:
        #   self.metric_writer.add_text("Train", "Training finished.")
        #   self.metric_writer.close()


# This callback does not seem to work with `utils.primary_process_only`.
class TrainStateCheckpoint(Callback):
    """Callback that periodically saves train state checkpoints."""

    def __init__(
        self,
        base_dir: str,
        folder_prefix: str = "checkpoints",
        train_state_field: str = "default",
        save_every_n_step: int = 1000,
        world_size: int = 1,
        local_rank: int = -1,
    ):
        self.save_dir = os.path.join(base_dir, folder_prefix)
        self.train_state_field = train_state_field
        self.save_every_n_steps = save_every_n_step
        self.last_eval_metric = {}
        self.world_size = world_size
        self.local_rank = local_rank

        os.makedirs(self.save_dir, exist_ok=True)

    def on_train_begin(self, trainer: Trainer) -> None:
        """Sets up directory, saves initial or restore the most recent state."""
        # retrieve from existing checkpoints if possible
        checkpoint_path = self._get_latest_checkpoint()
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, weights_only=True)

            model_compiled = (
                trainer.is_compiled
            )  # check whether current model is compiled
            model_ddp = (
                trainer.is_parallelized
            )  # check whether current model is trained parallelized
            checkpoint_compiled = checkpoint[
                "is_compiled"
            ]  # check if stored model was compiled
            checkpoint_ddp = checkpoint[
                "is_parallelized"
            ]  # check if stored model was trained in parallel

            keyword_compiled = "_orig_mod."
            keyword_ddp = "module."

            if not model_compiled and checkpoint_compiled:
                # stored model was compiled, current model is not: delete _orig_mod. in every key
                checkpoint["model_state_dict"] = {
                    key.replace(keyword_compiled, ""): value
                    for key, value in checkpoint["model_state_dict"].items()
                }

            if model_compiled and not checkpoint_compiled:
                # stored model not compiled, current model is compiled: add _orig_mod. at the beginning
                checkpoint["model_state_dict"] = {
                    keyword_compiled + key: value
                    for key, value in checkpoint["model_state_dict"].items()
                }

            if not model_ddp and checkpoint_ddp:
                # stored model trained in parallel but current model is not
                checkpoint["model_state_dict"] = {
                    key.replace(keyword_ddp, ""): value
                    for key, value in checkpoint["model_state_dict"].items()
                }

            if model_ddp and not checkpoint_ddp:
                # stored model was not trained in parallel but current model is
                checkpoint["model_state_dict"] = {
                    keyword_ddp + key: value
                    for key, value in checkpoint["model_state_dict"].items()
                }

            # Load stored states
            trainer.model.denoiser.load_state_dict(checkpoint["model_state_dict"])
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            trainer.train_state.step = checkpoint["step"]
            if (self.world_size > 1 and self.local_rank == 0) or self.world_size == 1:
                print("Continue Training from Checkpoint")

    def on_train_batches_end(
        self, trainer: Trainer, train_metrics: ComputedMetrics
    ) -> None:
        """Save checkpoints periodically after training batches"""
        cur_step = trainer.train_state.step

        if cur_step % self.save_every_n_steps == 0:
            if (self.world_size > 1 and self.local_rank == 0) or self.world_size == 1:
                self._save_checkpoint(trainer, cur_step, train_metrics)

    def on_eval_batches_end(
        self, trainer: Trainer, eval_metrics: ComputedMetrics
    ) -> None:
        """Store the evaluation metrics for inclusion in checkpoints"""
        self.last_eval_metric = eval_metrics

    def on_train_end(self, trainer: Trainer) -> None:
        """Save a final checkpoint at the end of training"""
        cur_step = trainer.train_state.step
        if (self.world_size > 1 and self.local_rank == 0) or self.world_size == 1:
            self._save_checkpoint(trainer, cur_step, self.last_eval_metric, force=True)

    def _save_checkpoint(
        self, trainer: Trainer, step: int, metrics: ComputedMetrics, force: bool = False
    ) -> None:
        """Internal method to handle checkpoint saving."""
        checkpoint = {
            "model_state_dict": trainer.model.denoiser.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "ema_param": trainer.train_state.ema if trainer.store_ema else None,
            "step": step,
            "metrics": metrics,
            "is_compiled": trainer.is_compiled,
            "is_parallelized": trainer.is_parallelized,
        }
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_{step}.pth")
        torch.save(checkpoint, checkpoint_path)

    def _get_latest_checkpoint(self) -> Optional[str]:
        """Retrieve the path to the latest checkpoint if available."""
        checkpoints = [f for f in os.listdir(self.save_dir) if f.endswith(".pth")]
        if not checkpoints:
            return None

        checkpoints.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]))
        return os.path.join(self.save_dir, checkpoints[-1])


class TqdmProgressBar(Callback):
    """Tqdm progress bar callback to monitor training progress in real time."""

    def __init__(
        self,
        total_train_steps: int | None,
        train_monitors: Sequence[str],
        eval_monitors: Sequence[str] = (),
        world_size: int = 1,
        local_rank: int = -1,
    ):
        """ProgressBar constructor.

        Args:
          total_train_steps: the total number of training steps, which is displayed
            as the maximum progress on the bar.
          train_monitors: keys in the training metrics whose values are updated on
            the progress bar after every training metric aggregation.
          eval_monitors: same as `train_monitors` except applying to evaluation.
        """
        super().__init__()
        self.total_train_steps = total_train_steps
        self.train_monitors = train_monitors
        self.eval_monitors = eval_monitors
        self.current_step = 0
        self.eval_postfix = {}  # keeps record of the most recent eval monitor
        self.bar = None
        self.world_size = world_size
        self.local_rank = local_rank

    def on_train_begin(self, trainer: Trainer) -> None:
        del trainer
        if (self.world_size > 1 and self.local_rank == 0) or self.world_size == 1:
            self.bar = tqdm.tqdm(total=self.total_train_steps, unit="step")

    def on_train_batches_end(
        self, trainer: Trainer, train_metrics: ComputedMetrics
    ) -> None:
        if (self.world_size > 1 and self.local_rank == 0) or self.world_size == 1:
            assert self.bar is not None
            self.bar.update(trainer.train_state.step - self.current_step)
            self.current_step = trainer.train_state.step
            postfix = {
                monitor: train_metrics[monitor] for monitor in self.train_monitors
            }
            self.bar.set_postfix(**postfix, **self.eval_postfix)

    def on_eval_batches_end(
        self, trainer: Trainer, eval_metrics: ComputedMetrics
    ) -> None:
        del trainer
        self.eval_postfix = {
            monitor: eval_metrics[monitor].item() for monitor in self.eval_monitors
        }

    def on_train_end(self, trainer: Trainer) -> None:
        del trainer
        if (self.world_size > 1 and self.local_rank == 0) or self.world_size == 1:
            assert self.bar is not None
            self.bar.close()
