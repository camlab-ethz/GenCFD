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

import unittest
import torch

from model.building_blocks.layers.axial_attention import AddAxialPositionEmbedding, AxialSelfAttention

SEED = 0
RNG = torch.Generator().manual_seed(SEED)

class AxialAttentionLayersTest(unittest.TestCase):

  def test_self_attn_output_shape(self):
    test_cases = [
      ((8, 8), -2), ((8, 8, 8), -2), ((8, 8, 8), -3)
    ]
    for test_case in test_cases:
        input_shape, axis = test_case

        inputs = torch.ones(input_shape)
        model = AxialSelfAttention(
            num_heads=4,
            rng=RNG,
            attention_axis=axis,
        )
        out = model(inputs=inputs)
        self.assertEqual(out.shape, input_shape)


  def test_pos_embedding_output_shape(self):
    test_cases = [
      ((8, 8), -2), ((8, 8, 8), -2), ((8, 8, 8), -3)
    ]
    for test_case in test_cases:
        input_shape, axis = test_case

        inputs = torch.ones(input_shape)
        model = AddAxialPositionEmbedding(position_axis=axis)
        out = model(inputs=inputs)
        self.assertEqual(out.shape, input_shape)


if __name__ == "__main__":
  unittest.main()
