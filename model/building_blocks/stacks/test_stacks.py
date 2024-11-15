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
import unittest
from model.building_blocks.stacks.ustacks import UStack, UpsampleFourierGaussian
from utils.model_utils import reshape_jax_torch
from model.building_blocks.stacks.dtstack import DStack
from model.building_blocks.stacks.dstack_3d import DStack as DStack3D
from model.building_blocks.stacks.ustack_3d import UStack as UStack3D

SEED = 0
RNG = torch.manual_seed(SEED)

class UStackTest(unittest.TestCase):

    def test_ustack(self):
        test_cases = [
            ((1, 2, 12), (1, 128), [(1, 16, 96), (1, 8, 48), 
                                    (1, 8, 48), (1, 4, 24), (1, 4, 24), 
                                    (1, 2, 12), (1, 2, 12)], (2, 2, 2), (1, 2, 28)),
            ((1, 2, 2, 12), (1, 28), [(1, 16, 16, 96), (1, 8, 8, 48), 
                                      (1, 8, 8, 48), (1, 4, 4, 24), (1, 4, 4, 24), 
                                      (1, 2, 2, 12), (1, 2, 2, 12)], (2, 2, 2), (1, 2, 10, 28))
        ]
        for inp_shape, emb_shape, skip_shape_list, num_res_blocks, new_shape in test_cases:
            with self.subTest(inp_shape=inp_shape, emb_shape=emb_shape, 
                              skip_shape_list=skip_shape_list,
                              num_res_blocks=num_res_blocks):
                inputs = torch.randn(inp_shape)
                skips = [torch.randn(i) for i in skip_shape_list]
                emb = torch.randn(emb_shape)

                model_ustack = UStack((12, 8, 4), num_res_blocks, num_res_blocks, rng=RNG)
                out = model_ustack(inputs, emb, skips, True)

                model_fg = UpsampleFourierGaussian(new_shape, num_res_blocks, 16, inputs.shape[1], rng=RNG)
                out_fgup, out_fg = model_fg(inputs, emb, True)
    

class DStackTest(unittest.TestCase):

    def test_dstack(self):
        test_cases = [
            ((1, 259, 16), (1, 128), (2, 2, 2), (4, 8, 12)),
            ((1, 259, 16, 16), (1, 128), (2, 2, 2), (4, 8, 12)),
            ((2, 3, 64), (2, 128), (2, 2, 2), (4, 8, 12)),
            ((2, 3, 64, 64), (2, 128), (2, 2, 2), (4, 8, 12))
        ]
        for inp_shape, emb_shape, down_shape, num_channels in test_cases:
            with self.subTest(
                inp_shape=inp_shape, emb_shape=emb_shape, 
                down_shape=down_shape, num_channels=num_channels 
            ):
                inputs = torch.randn(inp_shape)
                emb = torch.randn(emb_shape)
                num_res_blocks = down_shape
                downsample_ratio = down_shape
                model = DStack(
                    num_channels, num_res_blocks, downsample_ratio, rng=RNG, use_attention=True, num_heads=4
                    )
                out = model(inputs, emb, is_training=True)


class DStack3DTest(unittest.TestCase):

    def test_dstack(self):
        test_cases = [
            # ((1, 259, 16), (1, 128), (2, 2, 2), (4, 8, 12)),
            # ((1, 259, 16, 16), (1, 128), (2, 2, 2), (4, 8, 12)),
            # ((2, 3, 64), (2, 128), (2, 2, 2), (4, 8, 12)),
            # ((2, 3, 64, 64), (2, 128), (2, 2, 2), (4, 8, 12)),
            ((2, 3, 64, 64, 64), (2, 128), (2, 2, 2), (4, 8, 12))
        ]
        for inp_shape, emb_shape, down_shape, num_channels in test_cases:
            with self.subTest(
                inp_shape=inp_shape, emb_shape=emb_shape, 
                down_shape=down_shape, num_channels=num_channels 
            ):
                inputs = torch.randn(inp_shape)
                emb = torch.randn(emb_shape)
                num_res_blocks = down_shape
                downsample_ratio = down_shape
                use_spatial_attention = (False, False, True)
                model = DStack3D(
                    num_channels, 
                    num_res_blocks, 
                    downsample_ratio, 
                    use_spatial_attention=use_spatial_attention,
                    rng=RNG, 
                    num_heads=4,
                    use_positional_encoding=True
                )
                out = model(inputs, emb, is_training=True)



class UStack3DTest(unittest.TestCase):

    def test_ustack(self):
        test_cases = [
            ((1, 2, 2, 2, 12), (1, 128), [(1, 16, 16, 16, 96), (1, 8, 8, 8, 48), 
                                    (1, 8, 8, 8, 48), (1, 4, 4, 4, 24), (1, 4, 4, 4, 24), 
                                    (1, 2, 2, 2, 12), (1, 2, 2, 2, 12)], (2, 2, 2), (1, 2, 10, 28))
        ]
        for inp_shape, emb_shape, skip_shape_list, num_res_blocks, new_shape in test_cases:
            with self.subTest(inp_shape=inp_shape, emb_shape=emb_shape, 
                              skip_shape_list=skip_shape_list,
                              num_res_blocks=num_res_blocks):
                inputs = torch.randn(inp_shape)
                skips = [torch.randn(i) for i in skip_shape_list]
                emb = torch.randn(emb_shape)

                model_ustack = UStack3D(
                    (12, 8, 4), 
                    num_res_blocks, 
                    num_res_blocks, 
                    (False, False, True), 
                    rng=RNG, 
                    num_heads=4    
                )
                out = model_ustack(inputs, emb, skips, True)
    

if __name__ == "__main__":
    unittest.main()

