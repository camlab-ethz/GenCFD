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

from train import trainers

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
      save_every_n_step: int = 1000
  ):
    self.save_dir = os.path.join(base_dir, folder_prefix)
    self.train_state_field = train_state_field
    self.save_every_n_steps = save_every_n_step
    self.last_eval_metric = {}

    os.makedirs(self.save_dir, exist_ok=True)

  def on_train_begin(self, trainer: Trainer) -> None:
    """Sets up directory, saves initial or restore the most recent state."""
    # retrieve from existing checkpoints if possible
    checkpoint_path = self._get_latest_checkpoint()
    if checkpoint_path:
      checkpoint = torch.load(checkpoint_path, weights_only=True)
      
      is_compiled = checkpoint['is_compiled'] # check if stored model was compiled
      if is_compiled:
        # stored model was compiled, thus the keys are stored with _orig_mod. and needs to be 
        checkpoint['model_state_dict'] = {
          key.replace('_orig_mod.', ''): value for key, value in checkpoint['model_state_dict'].items()
        }

      # Load stored states
      trainer.model.denoiser.load_state_dict(checkpoint['model_state_dict'])
      trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      trainer.train_state.step = checkpoint['step']
      print("Continue Training from Checkpoint")
    
    # Check if model should be compiled for faster training
    self.compile_model(trainer)
  
  def compile_model(self, trainer: Trainer) -> None:
    """Model should be compiled!"""
    if trainer.is_compiled:
      print(f"Compile Model for Faster Training")
      # set model first to train to avoid switches between train and eval
      trainer.model.denoiser.train() 
      # Compile model to speedup training time
      compiled_denoiser = torch.compile(trainer.model.denoiser)
      # replace denoiser in frozen dataclass
      object.__setattr__(trainer.model, 'denoiser', compiled_denoiser)

  def on_train_batches_end(
      self, trainer: Trainer, train_metrics: ComputedMetrics
  ) -> None:
    """Save checkpoints periodically after training batches"""
    cur_step = trainer.train_state.step

    if cur_step % self.save_every_n_steps == 0:
      self._save_checkpoint(trainer, cur_step, train_metrics)

  def on_eval_batches_end(
      self, trainer: Trainer, eval_metrics: ComputedMetrics
  ) -> None:
    """Store the evaluation metrics for inclusion in checkpoints"""
    self.last_eval_metric = eval_metrics

  def on_train_end(self, trainer: Trainer) -> None:
    """Save a final checkpoint at the end of training"""
    cur_step = trainer.train_state.step
    self._save_checkpoint(trainer, cur_step, self.last_eval_metric, force=True)

  def _save_checkpoint(self, 
                       trainer: Trainer, 
                       step: int, 
                       metrics: ComputedMetrics, 
                       force: bool = False
                       ) -> None:
    """Internal method to handle checkpoint saving."""
    checkpoint = {
      'model_state_dict': trainer.model.denoiser.state_dict(),
      'optimizer_state_dict': trainer.optimizer.state_dict(),
      'step': step,
      'metrics': metrics,
      'is_compiled': trainer.is_compiled
    }
    checkpoint_path = os.path.join(self.save_dir, f"checkpoint_{step}.pth")
    torch.save(checkpoint, checkpoint_path)

  def _get_latest_checkpoint(self) -> Optional[str]:
    """Retrieve the path to the latest checkpoint if available."""
    checkpoints = [f for f in os.listdir(self.save_dir) if f.endswith(".pth")]
    if not checkpoints:
      return None
    
    checkpoints.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
    return os.path.join(self.save_dir, checkpoints[-1])


class TqdmProgressBar(Callback):
  """Tqdm progress bar callback to monitor training progress in real time."""

  def __init__(
      self,
      total_train_steps: int | None,
      train_monitors: Sequence[str],
      eval_monitors: Sequence[str] = (),
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

  def on_train_begin(self, trainer: Trainer) -> None:
    del trainer
    self.bar = tqdm.tqdm(total=self.total_train_steps, unit="step")

  def on_train_batches_end(
      self, trainer: Trainer, train_metrics: ComputedMetrics
  ) -> None:
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
    assert self.bar is not None
    self.bar.close()