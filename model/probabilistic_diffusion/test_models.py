import unittest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import dataloader
from diffusion import diffusion as dfn_lib

from model.building_blocks.unets.unets import UNet
from model.probabilistic_diffusion.denoising_model import DenoisingModel
from train.trainers import DenoisingTrainer

class TestDenoisingModel(unittest.TestCase):
    def test_denoiser(self):
        test_cases = [
            ((64, 64), "CIRCULAR", (2, 2, 2), False),
        ]

        for spatial_dims, padding_method, ds_ratio, hr_res in test_cases:
            with self.subTest(
                spatial_dims=spatial_dims, padding_method=padding_method,
                ds_ratio=ds_ratio, hr_res=hr_res
            ):
                batch, channels = 2, 3
                x = torch.randn((batch, channels, *spatial_dims))
                sigma = torch.linspace(0, 1, batch)
                
                denoiser_model = UNet(
                    out_channels=channels,
                    num_channels=(4, 8, 12),
                    downsample_ratio=ds_ratio,
                    num_blocks=2,
                    padding_method=padding_method,
                    num_heads=4,
                    use_position_encoding=True,
                    use_hr_residual=hr_res,
                )

                # out = denoiser_model.forward(x, sigma, is_training=True)
                # self.assertEqual(out.shape, x.shape)

                DATA_STD = 0.31
                diffusion_scheme = dfn_lib.Diffusion.create_variance_exploding(
                    sigma=dfn_lib.tangent_noise_schedule(),
                    data_std=DATA_STD,
                )

                model = DenoisingModel(
                    input_shape=x.shape,
                    denoiser=denoiser_model,
                    noise_sampling=dfn_lib.log_uniform_sampling(
                        diffusion_scheme, clip_min=1e-4, uniform_grid=True,
                    ),
                    noise_weighting=dfn_lib.edm_weighting(data_std=DATA_STD),
                )

                # trainer = DenoisingTrainer()

