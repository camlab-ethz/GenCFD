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