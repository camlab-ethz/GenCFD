import unittest
import torch
import numpy as np
import functools
from diffusion import diffusion, samplers, schedulers
from model.building_blocks.unets import unets
from solvers import sde

RNG = torch.Generator().manual_seed(0)

class TestTransform:

    def __call__(self, denoise_fn, guidance_inputs):
        return lambda x, t, cond: denoise_fn(x, t, cond) + guidance_inputs["const"]


class SamplersTest(unittest.TestCase):

    def test_sampler_output_shape(self):
        input_shape = (5, 1)
        num_samples = 4
        num_steps = 8
        sigma_schedule = diffusion.tangent_noise_schedule()
        scheme = diffusion.Diffusion.create_variance_exploding(sigma_schedule)
        solver = sde.EulerMaruyama()

        sampler = samplers.SdeSampler(
            input_shape=input_shape,
            integrator=solver,
            tspan=schedulers.exponential_noise_decay(scheme, num_steps),
            scheme=scheme,
            denoise_fn=lambda x, t, params: x,
            return_full_paths=True,
        )
        sample_paths = sampler.generate(num_samples=num_samples)
        self.assertEqual(sample_paths.shape, (num_steps + 1, num_samples) + input_shape)

    # def test_unet_denoiser(self):
    #     input_shape = (64, 64, 3)
    #     num_samples = 2
    #     num_steps = 4
    #     sigma_schedule = diffusion.tangent_noise_schedule()
    #     scheme = diffusion.Diffusion.create_variance_exploding(sigma_schedule)

    #     unet = unets.PreconditionedDenoiser(
    #         out_channels=input_shape[-1],
    #         rng=RNG,
    #         num_channels=(4, 8, 12),
    #         downsample_ratio=(2, 2, 2),
    #         num_blocks=2,
    #         num_heads=4,
    #         sigma_data=1.0,
    #         use_position_encoding=False,
    #     )
    #     # dummy_initialization:
    #     is_training = False
    #     x = torch.ones((1,) + input_shape, dtype=torch.float32)
    #     sigma = torch.tensor(1.0, dtype=torch.float32)

    #     variables = unet(x, sigma, is_training=is_training)

    #     denoise_fn = functools.partial(unet.forward, variables, is_training=False)

    #     sampler = samplers.SdeSampler(
    #         input_shape=input_shape,
    #         integrator=sde.EulerMaruyama(rng=RNG),
    #         tspan=schedulers.exponential_noise_decay(scheme, num_steps),
    #         scheme=scheme,
    #         denoise_fn=denoise_fn,
    #         guidance_transforms=(),
    #         return_full_paths=True,
    #     )

    #     samples_paths = sampler.generate(num_samples=num_samples)
    #     self.assertEqual(samples_paths.shape, (num_steps + 1, num_samples) + input_shape)


    # def test_output_shape_with_guidance(self):
    #     input_shape = (5, 1)
    #     num_samples = 4
    #     num_steps = 8
    #     sigma_schedule = diffusion.tangent_noise_schedule()
    #     scheme = diffusion.Diffusion.create_variance_exploding(sigma_schedule)

    #     sampler = samplers.OdeSampler(
    #         input_shape=input_shape,
    #         integrator=ode.HeunsMethod(),
    #         tspan=samplers.exponential_noise_decay(scheme, num_steps),
    #         scheme=scheme,
    #         denoise_fn=lambda x, t, cond: x * t,
    #         guidance_transforms=(TestTransform(),),
    #     )

    #     rng = torch.Generator()
    #     rng.manual_seed(0)
    #     samples = sampler.generate(num_samples=num_samples, rng=rng, guidance_inputs={"const": torch.ones(input_shape)})
    #     self.assertEqual(samples.shape, (num_samples,) + input_shape)

if __name__ == "__main__":
    unittest.main()