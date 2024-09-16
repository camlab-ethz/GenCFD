# Copyright 2024 The swirl_dynamics Authors.
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

import unittest
import torch
import numpy as np
from convolutions import ConvLayer, DownsampleConv


class ConvLayersTest(unittest.TestCase):


  def test_latlon_conv_output_shape_and_equivariance(self):
    test_cases = [
      ((8, 4, 8, 8), "latlon"), 
      # ((8, 8, 8, 4, 8), "lonlat"), # TODO: 3D not yet implemented!
    ]

    for input_shape, padding in test_cases:
        num_features = 6
        in_channels = input_shape[-1]
        inputs = torch.rand(input_shape).permute(0,3,2,1)
        model = ConvLayer(
            features=num_features, 
            padding=padding, 
            kernel_size=(3, 3),
            in_channels=in_channels,
        )
        out = model(inputs).permute(0,3,2,1)
        self.assertEqual(out.shape, input_shape[:-1] + (num_features,))

        # Test equivariance in the longitudinal direction.
        lon_axis = -1 if padding == "lonlat" else -2
        rolled_inputs = torch.roll(inputs, shifts=3, dims=lon_axis)
        out_ = model(rolled_inputs).permute(0,3,2,1)
        np.testing.assert_allclose(
            torch.roll(out, shifts=3, dims=lon_axis).detach().numpy(), 
            out_.detach().numpy(), 
            atol=1e-6
        )


  def test_downsample_conv_output_shape(self):
    test_cases = [
      ((8, 8, 8, 8), (2, 2), (8, 4, 4, 8)),
      ((8, 8, 8, 8, 8), (2, 2, 2), (8, 4, 4, 4, 6)),
    ]

    for input_shape, ratios, expected_output_shape in test_cases:
        in_channels = input_shape[-1]
        num_features = 6
        inputs = torch.ones(input_shape)
        model = DownsampleConv(in_channels=in_channels, features=num_features, ratios=ratios)

        if len(input_shape) == 4:
          out = model(inputs.permute(0,3,2,1)).permute(0,3,2,1)
        elif len(input_shape) == 5:
           out = model(inputs.permute(0,4,3,2,1)).permute(0,4,3,2,1)

        self.assertEqual(out.shape, expected_output_shape[:-1] + (num_features,))


if __name__ == "__main__":
  unittest.main()
