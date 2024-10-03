# Copyright 2024 The Cam Lab.
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
from multihead_attention import MultiHeadDotProductAttention

SEED = 0
RNG = torch.manual_seed(SEED)

class MultiHeadAttentionTest(unittest.TestCase):

    def test_multi_head_attention(self):
        test_cases = [
            ((2, 3, 4), 1), 
            ((5, 6, 8), 2), 
            ((8, 9, 15), 3)
        ]
        for inp_shape, nheads in test_cases:
            with self.subTest(inp_shape=inp_shape, nheads=nheads):
                inputs = torch.randn(inp_shape)
                model = MultiHeadDotProductAttention(
                    inputs.shape[-1], nheads, rng=RNG, normalize_qk=True
                    )
                out = model(inputs, inputs, inputs)

if __name__ == "__main__":
    unittest.main()