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

"""Training callback library."""

from collections.abc import Mapping, Sequence
import os
import time
from typing import Any, Optional

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from train import train_states, trainers
from utils import train_utils

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
      # options: ocp.CheckpointManagerOptions | None = None,
      save_every_n_step: int = 1000
  ):
    self.save_dir = os.path.join(base_dir, folder_prefix)
    self.train_state_field = train_state_field
    # self.ckpt_manager = ocp.CheckpointManager(
    #     self.save_dir,
    #     item_handlers={self.train_state_field: ocp.StandardCheckpointHandler()},
    #     options=options,
    # )
    self.save_every_n_steps = save_every_n_step
    self.last_eval_metric = {}

    os.makedirs(self.save_dir, exist_ok=True)

  def on_train_begin(self, trainer: Trainer) -> None:
    """Sets up directory, saves initial or restore the most recent state."""
    # retrieve from existing checkpoints if possible
    checkpoint_path = self._get_latest_checkpoint()
    if checkpoint_path:
      checkpoint = torch.load(checkpoint_path, weights_only=True)
      trainer.model.denoiser.load_state_dict(checkpoint['model_state_dict'])
      trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      trainer.train_state.step = checkpoint['step']


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
      'metrics': metrics
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


# @train_utils.primary_process_only
# class ProgressReport(Callback):
#   """Callback that reports progress during training.

#   Wraps `clu.periodic_actions.ReportProgress`, which reports progress summary
#   via `platform.work_unit().set_notes()`, and logs two additional timing
#   metrics, namely `uptime` and `steps_per_sec`.
#   """

#   def __init__(
#       self,
#       num_train_steps: int | None = None,
#       every_steps: int | None = None,
#       every_secs: float | None = 60.0,
#   ):
#     self.num_train_steps = num_train_steps
#     self.every_steps = every_steps
#     self.every_secs = every_secs
#     self.report_progress = None

#   def on_train_begin(self, trainer: Trainer) -> None:
#     del trainer
#     self.report_progress = periodic_actions.ReportProgress(
#         num_train_steps=self.num_train_steps,
#         every_steps=self.every_steps,
#         every_secs=self.every_secs,
#         writer=self.metric_writer,
#     )

#   def on_train_batches_end(
#       self, trainer: Trainer, train_metrics: ComputedMetrics
#   ) -> None:
#     del train_metrics
#     assert self.report_progress is not None
#     self.report_progress(trainer.train_state.int_step, time.time())


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


# @train_utils.primary_process_only
# class LogGinConfig(Callback):
#   """Write gin config string as text in TensorBoard."""

#   def on_train_begin(self, trainer: trainers.BaseTrainer) -> None:
#     config_str = gin.operative_config_str()
#     self.metric_writer.write_texts(
#         0, {"config": gin.markdown(config_str), "raw_config_str": config_str}
#     )
#     self.metric_writer.flush()


# def _get_markdown_param_table(
#     params: dict[str, np.ndTensor] | Mapping[str, Mapping[str, Any]],
# ) -> str:
#   """Returns a markdown table of parameters."""
#   param_table = parameter_overview.get_parameter_overview(
#       params, include_stats="global"
#   )
#   # Changes: remove first and second last rows (upper and lower table borders)
#   # and replace `+`s in 3rd row with `|`s.
#   rows = param_table.split("\n")
#   header = rows[1]
#   hline = rows[2].replace("+", "|")
#   body = rows[3:-2]
#   total = rows[-1]
#   return "\n".join([header, hline] + body + ["", total])


# @train_utils.primary_process_only
# @dataclasses.dataclass
# class ParameterOverview(Callback):
#   """Writes parameter overview to INFO log and/or TensorBoard.

#   Attributes:
#     log_to_info: Whether to print parameter overview to log (INFO level).
#     log_to_tb: Whether to add parameter overview to tensorboard (under text
#       tab).
#   """

#   log_to_info: bool = True
#   log_to_tb: bool = True

#   def on_train_begin(self, trainer: trainers.BaseTrainer) -> None:
#     train_state = trainer.unreplicated_train_state
#     if isinstance(train_state, train_states.BasicTrainState):
#       params = train_state.params
#       if self.log_to_info:
#         logging.info("Logging parameter overview.")
#         parameter_overview.log_parameter_overview(params)

#       if self.log_to_tb:
#         self.metric_writer.write_texts(
#             0,
#             {"parameters": _get_markdown_param_table(params)},
#         )
#         self.metric_writer.flush()
#     else:
#       logging.warning(
#           "ParameterOverview callback: unable extract parameters for overivew."
#       )


# Figures = Sequence[plt.Figure] | plt.Figure


# def figure_to_image(figures: Figures) -> np.ndTensor:
#   """Converts a sequence of figures to image data ingestable by tensorboard."""

#   # This import is rather heavy so we only do it when it's actually needed.
#   import matplotlib.backends.backend_agg as mpl_agg  # pylint: disable=g-import-not-at-top

#   def render_to_rgb(figure: plt.Figure) -> np.ndTensor:
#     canvas = mpl_agg.FigureCanvasAgg(figure)
#     canvas.draw()
#     data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
#     w, h = figure.canvas.get_width_height()
#     image_hwc = data.reshape([h, w, 4])[:, :, :3]
#     plt.close(figure)
#     return image_hwc

#   if isinstance(figures, plt.Figure):
#     figures = [figures]
#   images = [render_to_rgb(figure) for figure in figures]
#   return np.stack(images)


# class MatplotlibFigureAsImage(Callback):
#   """Makes matplotlib figures and writes to tensorboard.

#   Child classes should create model-specific plots in standard callback hooks
#   and call `.write_images()` to write them to TB. Plot data should be returned
#   from eval step of the model and declared as `CollectingMetric` in the trainer.

#   Pattern::

#     # Model
#     class Model:

#       def eval_step(...):
#         sample = ...  # Compute data required for plotting.
#         return {
#             "generated_sample": sample,
#         }

#     # Trainer
#     class Trainer:

#       @flax.struct.dataclass
#       class EvalMetrics(clu.metrics.Collection):
#         eval_plot_data: clu.metrics.CollectingMetric.from_outputs(
#             ("generated_sample",)  # Matches dict key in model.eval_step output.
#         )

#     # Custom plot callback
#     class PlotSamples(MplFigureAsImage):

#       def on_eval_batches_end(self, trainer, eval_metrics):
#         plot_data = eval_metrics["eval_plot_data"]  # Same name in trainer cls.
#         sample = plot_data["generated_sample"]  # Key in model.eval_step output.

#         # make plots
#         fig, ax = plt.subplots()
#         ax.imshow(sample[0])  # Plotting first element of aggregated samples.

#         # write plots to TB
#         self.write_images(
#             trainer.train_state.int_step, {"generated_sample": fig}
#         )
#   """

#   def write_images(self, step: int, images: Mapping[str, Figures]) -> None:
#     # Writing should only happen on primary host if metric writer is set up
#     # correctly (i.e. with `just_logging=jax.process_index() > 0`).
#     self.metric_writer.write_images(
#         step, {k: figure_to_image(v) for k, v in images.items()}
#     )


# @dataclasses.dataclass
# class LogLearningRateToTensorBoard(Callback):
#   """Logs the learning rate to tensorboard scalar.

#   Attributes:
#     lr_schedule: The (static) schedule which will be used to compute the
#       learning rate at given steps.
#   """

#   lr_schedule: optax.Schedule

#   def on_train_batches_end(
#       self, trainer: Trainer, train_metrics: ComputedMetrics
#   ) -> None:
#     self.metric_writer.write_scalars(
#         trainer.train_state.int_step,
#         {"learning_rate": self.lr_schedule(trainer.train_state.int_step)},
#     )


# @dataclasses.dataclass
# class InitializeFromCheckpoint(Callback):
#   """Initializes train state based on an existing checkpoint.

#   Before training starts, this callback loads a checkpoint and overrides
#   selected fields in the current train state with corresponding ones in the
#   checkpoint.

#   Note that this callback should always be passed before
#   `TrainStateCheckpoint` so that it does not interfere with the latter's ability
#   to restore training progress from unfinished runs.

#   Attributes:
#     checkpoint_dir: The directory containing the checkpoint to load.
#     step: The training step of the checkpoint to load. If `None`, the latest
#       step is loaded.
#     fields_to_override: Fields of the current train state to be overridden by
#       the corresponding ones (with the same name) in the checkpoint.
#   """

#   checkpoint_dir: str
#   step: int | None = None
#   fields_to_override: Sequence[str] = ("params",)

#   def on_train_begin(self, trainer: Trainer) -> None:
#     restored_state = trainer.train_state.restore_from_orbax_ckpt(
#         self.checkpoint_dir, step=self.step, ref_state=trainer.train_state
#     )
#     overrides = {
#         field: getattr(restored_state, field)
#         for field in self.fields_to_override
#     }
#     trainer.train_state = trainer.train_state.replace(**overrides)
