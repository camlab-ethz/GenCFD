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

import torch
from model.building_blocks.blocks.convolution_blocks import ResConv1x, ConvBlock, AdaptiveScale
from model.building_blocks.blocks.attention_block import AttentionBlock, AxialSelfAttentionBlock
import unittest

SEED = 0
RNG = torch.manual_seed(SEED)

class ResConv1xTest(unittest.TestCase):

    def test_ResConv1x(self):
        test_cases = [
            ((8, 8, 8, 8), (8, 6, 8, 8), 'True'),
            ((8, 8, 8, 8, 8), (8, 6, 8, 8, 8), 'True'),
            ((8, 8, 8, 8), (8, 8, 8, 8), 'False'),
            ((8, 8, 8, 8, 8), (8, 8, 8, 8, 8), 'False')
        ]
        for input_shape, expected_shape, project_skip in test_cases:
            with self.subTest(
                input_shape=input_shape,
                expected_shape=expected_shape,
                project_skip=project_skip
                ):
                input = torch.randint(0, 10, input_shape, dtype=torch.float32)
                model = ResConv1x(
                    input_shape[1], expected_shape[1], rng=RNG, project_skip=project_skip
                    )
                output = model(input)
                self.assertEqual(output.shape, expected_shape)

class AdaptiveScaleTest(unittest.TestCase):
    
    def test_AdaptiveScale(self):
        test_cases = [
            ((8, 8, 8, 8), (8, 8, 8, 8), (8, 10)),
            ((2, 3, 4, 5, 6), (2, 3, 4, 5, 6), (2, 100))
        ]
        for inp_shape, expected_shape, emb_shape in test_cases:
            with self.subTest(
                inp_shape=inp_shape, 
                expected_shape=expected_shape,
                emb_shape=emb_shape
                ):
                inputs = torch.randint(0, 10, inp_shape)
                emb = torch.randn(emb_shape, dtype=torch.float32)

                model = AdaptiveScale()
                out = model(inputs, emb)

                self.assertEqual(out.shape, expected_shape)

class ConvBlockTest(unittest.TestCase):

    def test_ConvBlock(self):
        test_cases = [
            ((8, 8, 8, 8), (8, 10, 8, 8), (8, 50), 'True'),
            # ((8, 8, 8, 8, 8), (8, 10, 8, 8, 8), (8, 50), 'True'),
            ((8, 8, 8, 8), (8, 10, 8, 8), (8, 50), 'False'),
            # ((8, 8, 8, 8, 8), (8, 10, 8, 8, 8), (8, 50), 'False')
        ]
        for inp_shape, expected_shape, emb_shape, is_training in test_cases:
            with self.subTest(
                inp_shape=inp_shape, 
                expected_shape=expected_shape, 
                emb_shape=emb_shape,
                is_training=is_training):
                inputs = torch.randn(inp_shape)
                emb = torch.randn(emb_shape)
                if len(inp_shape) == 4:
                    model = ConvBlock(
                        in_channels=inputs.shape[1],
                        out_channels=expected_shape[1], 
                        kernel_size=(3, 3), 
                        rng=RNG, 
                        dropout=0.1,
                        padding=1
                        )
                else:
                    model = ConvBlock(
                        in_channels=inputs.shape[1],
                        out_channels=expected_shape[1], 
                        kernel_size=(3, 3, 3), 
                        rng=RNG, 
                        dropout=0.1,
                        padding=1
                        )

                out = model(inputs, emb, is_training)

                self.assertEqual(out.shape, expected_shape)

class AttenionBlockTest(unittest.TestCase):

    def test_attention_block(self):
        test_cases = [
            (2, 4, 8), (4, 8, 16)
        ]
        for input_shape in test_cases:
            with self.subTest(input_shape=input_shape):
                inputs = torch.randn(input_shape)
                model = AttentionBlock(rng=RNG)
                out = model(inputs, False)
                self.assertEqual(out.shape, inputs.shape)

class AxialAttenionBlockTest(unittest.TestCase):

    def test_attention_block(self):
        test_cases = [
            (20, 3, 40, 40),
            (2, 4, 8), (4, 8, 16)
        ]
        for input_shape in test_cases:
            with self.subTest(input_shape=input_shape):
                inputs = torch.randn(input_shape)
                model = AxialSelfAttentionBlock(rng=RNG)
                out = model(inputs, False)
                self.assertEqual(out.shape, inputs.shape)
    
if __name__ == "__main__":
  unittest.main()