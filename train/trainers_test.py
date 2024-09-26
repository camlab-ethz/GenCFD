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

import unittest
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from unittest import mock
from model.base_model.base_model import BaseModel
from train.train_states import TrainState, BasicTrainState
from train.trainers import BaseTrainer, BasicTrainer

from torchmetrics import MetricCollection


# Mock batch generator
def dummy_iter(batch_sz):
  while True:
    yield {"1": torch.ones(batch_sz)}


class TestTrainer(BaseTrainer):
    class TrainMetrics:
        def __init__(self):
            self.train_loss = []

        def update(self, loss):
            if isinstance(loss, torch.Tensor):
                self.train_loss.append(loss.item())
            else:
                self.train_loss.append(loss)

        def compute(self):
            return {"train_loss": np.mean(self.train_loss)}

    class EvalMetrics:
        def __init__(self):
            self.eval_accuracy = []

        def update(self, accuracy):
            if isinstance(accuracy, torch.Tensor):
                self.eval_accuracy.append(accuracy.item())
            else:
                self.eval_accuracy.append(accuracy)

        def compute(self):
            return {"eval_accuracy": np.mean(self.eval_accuracy)}

    def initialize_train_state(self):
        return TrainState()

    def train_step(self, batch):
        pass  # to be mocked

    def eval_step(self, batch):
        pass  # to be mocked


class BaseTrainerTest(unittest.TestCase):

    def test_train(self):
        """Test training loop with mocked step function."""
        num_steps = 5
        rng = np.random.default_rng(42)
        test_train_losses = rng.uniform(size=(num_steps,))

        with mock.patch.object(TestTrainer, 'train_step', autospec=True) as mock_train_fn:
            # Mock the train_step function to return increasing step count and mocked metrics
            train_outputs = [
                (TrainState(step=i + 1), TestTrainer.TrainMetrics())
                for i, _ in enumerate(test_train_losses)
            ]

            for i, output in enumerate(train_outputs):
                output[1].update(torch.tensor(test_train_losses[i]))
            mock_train_fn.side_effect = train_outputs

            mock_model = mock.Mock(spec=BaseModel)
            # there is no mock method with train and to thus it's manually implemented!
            mock_model.to = mock.Mock(return_value=mock_model)
            mock_model.train = mock.Mock(return_value={"loss": torch.tensor(0.5)})
            # mock_model.train = mock.Mock(side_effect=lambda *args, **kwargs: {"loss": torch.tensor(0.5)})

            
            trainer = TestTrainer(model=mock_model, device=torch.device('cpu'))
            train_metrics = trainer.train(batch_iter=dummy_iter(1), num_steps=num_steps).compute()

        self.assertEqual(trainer.train_state.step, num_steps)
        self.assertTrue(np.allclose(train_metrics["train_loss"], np.mean(test_train_losses)))

    def test_eval(self):
        """Test evaluation loop with mocked step function."""
        num_steps = 10
        rng = np.random.default_rng(43)
        test_eval_accuracies = rng.uniform(size=(num_steps,))

        with mock.patch.object(TestTrainer, 'eval_step', autospec=True) as mock_eval_fn:
            # Mock the eval_step function to return mocked metrics
            eval_outputs = [
                TestTrainer.EvalMetrics() for _ in range(num_steps)
            ]

            for i, output in enumerate(eval_outputs):
                output.update(torch.tensor(test_eval_accuracies[i]))
            mock_eval_fn.side_effect = eval_outputs

            mock_model = mock.Mock(spec=BaseModel)
            mock_model.to = mock.Mock(return_value=mock_model)
            mock_model.eval = mock.Mock(return_value={"loss": torch.tensor(0.5)})

            trainer = TestTrainer(model=mock_model, device=torch.device('cpu'))
            eval_metrics = trainer.eval(batch_iter=dummy_iter(1), num_steps=num_steps).compute()

        self.assertEqual(trainer.train_state.step, 0)
        self.assertTrue(np.allclose(eval_metrics["eval_accuracy"], np.mean(test_eval_accuracies)))


if __name__ == '__main__':
    unittest.main()