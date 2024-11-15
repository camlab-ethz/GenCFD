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

"""Tests for utils."""

import tempfile
import os
import shutil
import unittest
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils import train_utils

abs_tol = 1e-6


class LoadTFeventsTest(unittest.TestCase):

  def test_load_scalars(self):
    steps = 10
    rng = np.random.default_rng(42)
    loss = rng.uniform(size=(steps,))

    with tempfile.TemporaryDirectory() as temp_dir:
      writer = SummaryWriter(log_dir=temp_dir)
      for s, l in enumerate(loss):
        writer.add_scalar("loss", l, s)

      writer.flush()
      writer.close()

      loaded = train_utils.load_scalars_from_tfevents(temp_dir)
      loaded_loss = [loaded[s]["loss"] for s in range(steps)]

      np.testing.assert_allclose(loaded_loss, loss, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
  unittest.main()
