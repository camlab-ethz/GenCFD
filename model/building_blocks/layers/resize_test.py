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
import numpy as np
from model.building_blocks.layers.resize import FilteredResize

SEED = 0
RNG = torch.Generator().manual_seed(SEED)


class ResizeLayersTest(unittest.TestCase):

  def test_filtered_resize_output_shape(self):
    test_cases = [
      ((5, 8, 7, 6), (7,), (5, 8, 7, 7)),
      ((4, 8, 6, 7, 5), (9, 9), (4, 8, 6, 9, 9)),
      ((4, 8, 6, 7, 5), (9, 9, 9), (4, 8, 9, 9, 9)),
      # ((4, 8, 6, 7, 5), (3, 3, 3, 3), (3, 8, 3, 3, 3)),
    ]
    for input_shape, output_size, expected_output_shape in test_cases:
      inputs = torch.ones(input_shape, dtype=torch.float32)
      model = FilteredResize(output_size=output_size, kernel_size=7, rng=RNG)
      out = model(inputs=inputs)
      self.assertEqual(out.shape, expected_output_shape)


if __name__ == "__main__":
  unittest.main()
