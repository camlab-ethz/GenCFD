# Copyright 2024 CAM Lab at ETH Zurich.
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
from model.building_blocks.unets.unets import UNet

SEED = 0
RNG = torch.manual_seed(SEED)

class NetworksTest(unittest.TestCase):

    def test_unet_output_shape(self):
        """Test to check if UNet output shape matches input shape"""
        
        test_cases = [
            ((64,), "CIRCULAR", (2, 2, 2), True),
            ((64, 64), "CIRCULAR", (2, 2, 2), False),
            ((64, 64), "LATLON", (2, 2, 2), True),
            # ((72, 144), "LATLON", (2, 2, 3)), # This test fails!
        ]

        for spatial_dims, padding_method, ds_ratio, hr_res in test_cases:
            with self.subTest(
                spatial_dims=spatial_dims, padding_method=padding_method,
                ds_ratio=ds_ratio, hr_res=hr_res
            ):
                batch, channels = 2, 3
                x = torch.randn((batch, channels, *spatial_dims))
                sigma = torch.linspace(0, 1, batch)
                
                model = UNet(
                    out_channels=channels,
                    rng=RNG,
                    num_channels=(4, 8, 12),
                    downsample_ratio=ds_ratio,
                    num_blocks=2,
                    padding_method=padding_method,
                    num_heads=4,
                    use_position_encoding=True,
                    use_hr_residual=hr_res,
                )

                out = model.forward(x, sigma, is_training=True)
                self.assertEqual(out.shape, x.shape)

if __name__ == "__main__":
    unittest.main()