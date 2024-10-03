import unittest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import dataloader
import diffusion as dfn_lib
from model.building_blocks.unets.unets import UNet

SEED = 0
RNG = torch.manual_seed(SEED)

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
                    rng=RNG,
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

                model = dfn_lib.DenoisingModel(
                    input_shape=x.shape[1:],
                    denoiser=denoiser_model,
                    noise_sampling=dfn_lib.log_uniform_sampling(
                        diffusion_scheme, clip_min=1e-4, uniform_grid=True,
                    ),
                    noise_weighting=dfn_lib.edm_weighting(data_std=DATA_STD),
                    rng=RNG,
                    seed=SEED
                )

                init_output = model.initialize()
                self.assertIsNotNone(init_output, "Model initialization failed!")

                batch_data = {
                    "lead_time": torch.ones((batch,)),
                    "data": x,
                }
                rng = torch.Generator().manual_seed(42)

                loss, (metric, _) = model.loss_fn(batch_data)
                self.assertTrue(loss.item() >= 0, "Loss should be non-negative.")
                self.assertIn("loss", metric, "Metric should contain 'loss' key.")

                eval_metrics = model.eval_fn({}, batch_data, rng)
                for key in eval_metrics.keys():
                    self.assertTrue(eval_metrics[key] >= 0, "Evaluation loss should be non-negative.")

if __name__=="__main__":
    unittest.main()
