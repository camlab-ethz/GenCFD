import unittest
import torch

from model.embeddings.position_emb import position_embedding

class PositionEmbeddingTest(unittest.TestCase):

    def test_position_embedding(self):
        test_cases = [
            (2, 10, 64), (2, 32, 64, 96), (2, 12, 24, 48, 72)
        ]
        for test_shape in test_cases:
            with self.subTest(test_shape=test_shape):
                inputs = torch.randn(test_shape)
                model = position_embedding(len(inputs.shape)-2)
                out = model(inputs)

if __name__=="__main__":
    unittest.main()