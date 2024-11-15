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

"""Tests for the reshape utilities used in the ViViT-based models."""
import unittest
import torch as th
import reshape_utils


class ReshapeTest(unittest.TestCase):

  def test_2d_to_1d_reshape_output_shape(self):
    test_cases = [
      ((2, 4, 8, 3), 1),
      ((2, 8, 8, 3), 1),
      ((2, 8, 8, 3), 2),
    ]
    for shape, axis in test_cases:
      with self.subTest(shape=shape, axis=axis):
        batch_size, height, width, channel = shape

        x = th.randn(shape)

        new_batch = (batch_size * width) if axis == 1 else (batch_size * height)
        new_token = height if axis == 1 else width

        out = reshape_utils.reshape_2d_to_1d_factorized(x, axis=axis)

        self.assertEqual(out.shape, (new_batch, new_token, channel))


  def test_3d_to_1d_reshape_output_shape(self):
    test_cases = [
      ((2, 4, 8, 2, 3), 1),
      ((2, 2, 8, 3, 1), 3),
      ((2, 8, 2, 8, 3), 2),
      ((2, 2, 8, 3, 1), 3),
      ((2, 8, 2, 8, 3), 1),
    ]
    for shape, axis in test_cases:
      with self.subTest(shape=shape, axis=axis):
        batch_size, time, height, width, channel = shape

        x = th.randn(shape)

        new_batch = {'1': batch_size * height * width,
                    '2': batch_size * time * width,
                    '3': batch_size * time * height}
        new_token = {'1': time,
                    '2': height,
                    '3': width}

        out = reshape_utils.reshape_3d_to_1d_factorized(x, axis=axis)

        self.assertEqual(
            out.shape, (new_batch[f'{axis}'], new_token[f'{axis}'], channel)
        )


  def test_indentity_cycle_reshape_2d(self):

    test_cases = [
      ((2, 4, 8, 3), 1),
      ((2, 2, 8, 3), 1),
      ((2, 8, 4, 3), 2),
    ]

    for shape, axis in test_cases:
      with self.subTest(shape=shape, axis=axis):

        x = th.randn(shape)

        out_reshaped = reshape_utils.reshape_2d_to_1d_factorized(x, axis=axis)
        out_identity = reshape_utils.reshape_to_2d_factorized(
            out_reshaped, axis=axis, two_d_shape=shape
        )

        self.assertEqual(out_identity.tolist(), x.tolist())


  def test_indentity_cycle_reshape_3d(self):
    test_cases = [
      ((2, 4, 8, 2, 3), 1),
      ((2, 2, 8, 3, 1), 3),
      ((2, 8, 2, 8, 3), 2),
      ((2, 2, 8, 3, 1), 3),
      ((2, 8, 2, 8, 3), 1),
    ]

    for shape, axis in test_cases:
      with self.subTest(shape=shape, axis=axis):
        x = th.randn(shape)

        out_reshaped = reshape_utils.reshape_3d_to_1d_factorized(x, axis=axis)
        out_identity = reshape_utils.reshape_to_3d_factorized(
            out_reshaped, axis=axis, three_d_shape=shape
        )

        self.assertEqual(out_identity.tolist(), x.tolist())


if __name__ == '__main__':
  unittest.main()
