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

import os
import unittest
from unittest import mock
from torch.utils.data import DataLoader, Dataset
import numpy as np
from train import train, train_states, trainers

MSGS = {
    "setup": "Train start",
    "train_begin": "Train batches begin",
    "train_end": "Train batches end",
    "eval_begin": "Eval batches begin",
    "eval_end": "Eval batches end",
    "end": "Train end",
}


def _mock_train_state(step):
  return train_states.BasicTrainState(
      step=step,
      params=mock.Mock(),
      opt_state=mock.Mock(),
      flax_mutables=mock.Mock(),
  )

class DummyCallback(callbacks.Callback):
  """A placeholder callback that does nothing."""
  def on_train_begin(self, trainer):
    pass

  def on_train_batches_begin(self, trainer):
    pass

  def on_train_batches_end(self, trainer, train_metrics):
    pass

  def on_eval_batches_begin(self, trainer):
    pass

  def on_eval_batches_end(self, trainer, eval_metrics):
    pass

  def on_train_end(self, trainer):
    pass

# class TestCallback(callbacks.Callback):
#   """Callback that writes various messages indicating callsite locations."""

#   def __init__(self, save_dir):
#     self.save_dir = save_dir
#     self.log_file = open(os.path.join(self.save_dir, "log.txt"), "w")

#   def on_train_begin(self, trainer):
#     self.log_file.write(MSGS["setup"] + "\n")

#   def on_train_batches_begin(self, trainer):
#     self.log_file.write(MSGS["train_begin"] + "\n")

#   def on_train_batches_end(self, trainer, train_metrics):
#     self.log_file.write(MSGS["train_end"] + "\n")

#   def on_eval_batches_begin(self, trainer):
#     self.log_file.write(MSGS["eval_begin"] + "\n")

#   def on_eval_batches_end(self, trainer, eval_metrics):
#     self.log_file.write(MSGS["eval_end"] + "\n")

#   def on_train_end(self, trainer):
#     self.log_file.write(MSGS["end"] + "\n")
#     self.log_file.close()


# def _expected_execution(train_steps, eval_period):
#   """Expected execution stages of `TestCallback`."""
#   d, r = divmod(train_steps, eval_period)
#   train_batch = ["train_begin", "train_end"]
#   eval_batch = ["eval_begin", "eval_end"]
#   lines = (
#       (train_batch * eval_period + eval_batch) * d
#       + train_batch * r
#       + eval_batch * int(r != 0)
#   )
#   return ["setup", *lines, "end"]

class DummyDataset(Dataset):
  """Simple dummy dataset for testing purposes."""
  def __init__(self, data_range):
    self.data = data_range

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]


class TrainTest(unittest.TestCase):
  def setUp(self):
    super().setUp()
    self.dummy_dataset = DummyDataset(list(range(1, 10)))
    self.dummy_dataloader = DataLoader(self.dummy_dataset, batch_size=1, shuffle=True)

    # Mock trainer with constant metrics returned
    self.test_trainer = mock.Mock(spec=trainers.BasicTrainer)
    self.test_trainer.train_state = _mock_train_state(0)
    train_metrics = mock.Mock()
    train_metrics.compute.return_value = {"loss": 0.1}
    self.test_trainer.train.return_value = train_metrics

    eval_metrics = mock.Mock()
    eval_metrics.compute.return_value = {"accuracy": 0.6}
    self.test_trainer.eval.return_value = eval_metrics

  def test_writes_train_metrics(self):
    """Test training iterations by examining metrics."""
    workdir = os.path.join(os.getcwd(), "temp_dir")
    os.makedirs(workdir, exist_ok=True)
    train.run(
        train_dataloader=self.dummy_dataloader,
        trainer=self.test_trainer,
        workdir=workdir,
        metric_aggregation_steps=2,
        total_train_steps=10,
    )
    written = utils.load_scalars_from_tfevents(workdir)
    self.assertEqual(len(written.keys()), np.ceil(10 / 2))
    self.assertTrue(all([step in written.keys() for step in [2, 4, 6, 8, 10]]))
    self.assertIn("loss", written[10].keys())

  def test_writes_eval_metrics(self):
    """Test evaluation iterations by examining metrics."""
    workdir = os.path.join(os.getcwd(), "temp_dir")
    os.makedirs(workdir, exist_ok=True)
    train.run(
        train_dataloader=self.dummy_dataloader,
        trainer=self.test_trainer,
        workdir=workdir,
        metric_aggregation_steps=2,
        total_train_steps=20,
        eval_dataloader=self.dummy_dataloader,
        eval_every_steps=6,
        num_batches_per_eval=1,
    )
    written = utils.load_scalars_from_tfevents(workdir)
    for step in np.arange(6, 20, 6):
        self.assertIn("loss", written[step].keys())
        self.assertIn("accuracy", written[step].keys())
    self.assertIn("accuracy", written[20].keys())

  def test_raises_eval_period_divisibility_error(self):
    """Test error when eval period is not divisible by aggregation steps."""
    with self.assertRaises(ValueError):
      train.run(
          train_dataloader=self.dummy_dataloader,
          trainer=self.test_trainer,
          workdir=os.path.join(os.getcwd(), "temp_dir"),
          metric_aggregation_steps=10,
          total_train_steps=100,
          eval_dataloader=self.dummy_dataloader,
          eval_every_steps=42,
          num_batches_per_eval=1,
      )

  def test_triggers_callbacks(self):
    """Test that callbacks are properly triggered."""
    workdir = os.path.join(os.getcwd(), "temp_dir")
    os.makedirs(workdir, exist_ok=True)
    train.run(
      train_dataloader=self.dummy_dataloader,
      trainer=self.test_trainer,
      workdir=workdir,
      metric_aggregation_steps=1,
      total_train_steps=5,
      eval_dataloader=self.dummy_dataloader,
      eval_every_steps=3,
      num_batches_per_eval=1,
      callbacks=[DummyCallback(workdir)],
    )
    expected_stages = _expected_execution(5, 3)
    with open(os.path.join(workdir, "log.txt")) as log_file:
      for stage in expected_stages:
        self.assertEqual(log_file.readline().rstrip(), MSGS[stage])


if __name__ == "__main__":
    unittest.main()