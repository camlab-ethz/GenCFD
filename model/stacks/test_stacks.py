import torch
import unittest
from model.stacks.ustack import UStack
from utils.model_utils import reshape_jax_torch

class UStackTest(unittest.TestCase):

    def test_ustack(self):
        test_cases = [
            ((1, 2, 12), (1, 128), [(1, 16, 96), (1, 8, 48), 
                                    (1, 8, 48), (1, 4, 24), (1, 4, 24), 
                                    (1, 2, 12), (1, 2, 12)], (2, 2, 2)),
            ((1, 2, 2, 12), (1, 28), [(1, 16, 16, 96), (1, 8, 8, 48), 
                                      (1, 8, 8, 48), (1, 4, 4, 24), (1, 4, 4, 24), 
                                      (1, 2, 2, 12), (1, 2, 2, 12)], (2, 2, 2))
        ]
        for inp_shape, emb_shape, skip_shape_list, num_res_blocks in test_cases:
            with self.subTest(inp_shape=inp_shape, emb_shape=emb_shape, 
                              skip_shape_list=skip_shape_list,
                              num_res_blocks=num_res_blocks):
                inputs = torch.randn(inp_shape)
                skips = [torch.randn(i) for i in skip_shape_list]
                emb = torch.randn(emb_shape)

                model = UStack((12, 8, 4), num_res_blocks, num_res_blocks)

                out = model(inputs, emb, skips, True)

if __name__ == "__main__":
    unittest.main()

