import torch
from model.blocks.adaptive_scaling import AdaptiveScale
from model.blocks.convolution_blocks import ResConv1x, ConvBlock
from model.blocks.attention_block import AttentionBlock
from utils.model_utils import reshape_jax_torch
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
                conv_kwargs = {'in_channels': inputs.shape[1], 'padding': 1}
                if len(inp_shape) == 4:
                    model = ConvBlock(expected_shape[1], (3, 3), dropout=0.1, **conv_kwargs)
                else:
                    model = ConvBlock(expected_shape[1], (3, 3, 3), dropout=0.1, **conv_kwargs)

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
                model = AttentionBlock()
                out = model(inputs, False)
    
if __name__ == "__main__":
  unittest.main()