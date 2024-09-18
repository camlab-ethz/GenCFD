import unittest
import torch
from multihead_attention import MultiHeadDotProductAttention

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
                model = MultiHeadDotProductAttention(inputs.shape[-1], nheads, normalize_qk=True)
                out = model(inputs, inputs, inputs)

if __name__ == "__main__":
    unittest.main()