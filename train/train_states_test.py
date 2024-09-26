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
import tempfile
import unittest
import torch
from torch import nn, optim
from train_states import TrainState, BasicTrainState

class TrainStateTest(unittest.TestCase):

    def test_create_state(self):
        state = TrainState(step=0)
        self.assertEqual(state.step.item(), 0)

    def test_int_step(self):
        state = TrainState(step=0)
        self.assertEqual(state.int_step, 0)

    def test_save_and_load_state_from_checkpoint(self):
        state = TrainState(step=100)
        save_dir = os.getcwd()
        ckpt_path = os.path.join(save_dir, "checkpoint.pth")

        state.save_checkpoint(ckpt_path)

        loaded_state = TrainState.restore_from_checkpoint(ckpt_path)
        self.assertEqual(loaded_state.int_step, 100)


class BasicTrainStateTest(unittest.TestCase):

    def test_save_and_load_state_from_checkpoint(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.dense = nn.Linear(5, 5)

        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        # Create a BasicTrainState
        state = BasicTrainState(
            model=model,
            optimizer=optimizer,
            step=torch.tensor(0),
            params=model.state_dict(),
            opt_state=optimizer.state_dict()
        )

        save_dir = os.getcwd()
        ckpt_path = os.path.join(save_dir, "checkpoint.pth")

        state.save_checkpoint(ckpt_path)

        loaded_state = BasicTrainState.restore_from_checkpoint(ckpt_path, model, optimizer)

        self.assertEqual(state.int_step, loaded_state.int_step)
        for key in state.params.keys():
            self.assertTrue(torch.equal(state.params[key], loaded_state.params[key]))
        self.assertEqual(state.opt_state, loaded_state.opt_state)


if __name__ == "__main__":
    unittest.main()
