import torch
import unittest
from model.building_blocks.stacks.ustacks import UStack, UpsampleFourierGaussian
from utils.model_utils import reshape_jax_torch
from model.building_blocks.stacks.dtstack import DStack

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

                model_ustack = UStack((12, 8, 4), num_res_blocks, num_res_blocks)
                out = model_ustack(inputs, emb, skips, True)

                model_fg = UpsampleFourierGaussian(new_shape, num_res_blocks, 16, inputs.shape[1])
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
                model = DStack(num_channels, num_res_blocks, downsample_ratio, use_attention=True, num_heads=4)
                out = model(inputs, emb, is_training=True)

if __name__ == "__main__":
    unittest.main()

