import torch
from model.diff_model.unets import ResConv1x, AdaptiveScale
import unittest

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
                model = ResConv1x(input_shape[1], expected_shape[1], project_skip=project_skip)
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
    
if __name__ == "__main__":
  unittest.main()