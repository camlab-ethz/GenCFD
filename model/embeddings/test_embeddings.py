# Copyright 2024 The swirl_dynamics Authors and CAM Lab at ETH Zurich.
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

from model.embeddings.position_emb import position_embedding
from model.embeddings.fourier_emb import FourierEmbedding

class PositionEmbeddingTest(unittest.TestCase):

    def test_position_embedding(self):
        test_cases = [
            (2, 10, 64), (2, 32, 64, 96), (2, 12, 24, 48, 72)
        ]
        for test_shape in test_cases:
            with self.subTest(test_shape=test_shape):
                inputs = torch.randn(test_shape)
                embedding = position_embedding(len(inputs.shape)-2)
                out = embedding(inputs)

class FourierEmbeddingTest(unittest.TestCase):

    def test_fourier_embedding(self):
        test_cases = [
            (1,), (2,), (3,)
        ]
        for inp_shape in test_cases:
            with self.subTest(inp_shape=inp_shape):
                inputs = torch.randn(inp_shape)
                embedding = FourierEmbedding()
                out = embedding(inputs)

if __name__=="__main__":
    unittest.main()