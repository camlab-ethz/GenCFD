import sys
import time

import os
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.utils.data import DataLoader

from train import train
import diffusion as dfn_lib
from dataloader.dataset import DataIC_Vel
from train.train_states import DenoisingModelTrainState
from solvers.sde import EulerMaruyama

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
RNG = torch.Generator(device=device)
RNG.manual_seed(SEED)

sys.path.append("/usr/local/cuda/bin/ptxas")


if __name__=="__main__":
    dataset = DataIC_Vel()

    # Workdirectory set manually
    cwd = os.getcwd()
    workdir = cwd + '/outputs'

    # Model parameters set manually
    train_batch_size = 5
    use_position_encoding = True
    num_head = 8
    noise_embed_dim = 128
    downsample_ratio = (2, 2)
    padding_method="circular"
    out_channels = 2
    num_channels = (64, 128)
    use_attention = True
    num_blocks = 4
    DATA_STD = 0.5
    ema_decay = 0.999
    peak_lr = 1e-4
    num_train_steps = 1_000_000 

    inp_shape = dataset.__getitem__(0).shape
    out_shape = (out_channels,) + tuple(inp_shape[1:])

    # Set model
    denoiser_model = dfn_lib.PreconditionedDenoiserUNet(
        out_channels=out_channels,
        rng=RNG,
        num_channels=num_channels,
        downsample_ratio=downsample_ratio,
        num_blocks=num_blocks,
        noise_embed_dim=noise_embed_dim,
        padding_method=padding_method,
        use_attention=use_attention,
        use_position_encoding=use_position_encoding,
        num_heads=num_head,
        device=device,
        sigma_data=DATA_STD
    )

    diffusion_scheme = dfn_lib.Diffusion.create_variance_exploding(
        sigma=dfn_lib.exponential_noise_schedule(device=device),
        data_std=DATA_STD,
        device=device
    )

    noise_sampling = dfn_lib.log_uniform_sampling(
        diffusion_scheme, clip_min=1e-4, uniform_grid=True, device=device
    )

    noise_weighting=dfn_lib.edm_weighting(data_std=DATA_STD, device=device)


    print(" ")
    print("Denoiser Initialization")

    # Determine the shape of the dataset:
    # sample = train_dataset.__getitem__(0)

    model = dfn_lib.DenoisingModel(
        input_shape=inp_shape,
        input_channel=out_channels,
        denoiser=denoiser_model,
        noise_sampling=noise_sampling,
        noise_weighting=noise_weighting,
        rng=RNG,
        device=device
    )

    # Dummy initialize the model:
    model.initialize(batch_size=train_batch_size)

    trainer = train.trainers.DenoisingTrainer(
        model=model,
        optimizer=optim.Adam(model.denoiser.parameters(), lr=peak_lr),
        ema_decay=ema_decay,
        device=device
    )

    print(" ")
    print("Sampler Initialization")

    trained_state = DenoisingModelTrainState.restore_from_checkpoint(
        f"{workdir}/checkpoints/checkpoint_15000.pth", model=model.denoiser, optimizer=trainer.optimizer
    )

    # Construct the inference function
    denoise_fn = trainer.inference_fn_from_state_dict(
        trained_state, use_ema=True, denoiser=denoiser_model, task='solver'
    )

    sampler = dfn_lib.SdeSampler(
        input_shape=out_shape,
        integrator=EulerMaruyama(rng=RNG),
        tspan=dfn_lib.edm_noise_decay(
            diffusion_scheme, rho=7, num_steps=2, end_sigma=1e-3, device=device
        ),
        scheme=diffusion_scheme,
        denoise_fn=denoise_fn,
        guidance_transforms=(),
        apply_denoise_at_end=True,
        return_full_paths=False,  # Set to `True` if the full sampling paths are needed
        device=device
    )

    i_max = train_batch_size
    result = []

    # generate = functools.partial(sampler.generate, num_samples=1)
    normalized_tiny_dataset = dataset.get_tiny_dataset(20)
    u0 = normalized_tiny_dataset[:,:out_channels, ...].to(device=device)
    u = normalized_tiny_dataset[:, out_channels:, ...].to(device=device)

    # Only necessary if num_samples > 1!!!
    # u0_norm_rep = u0.unsqueeze(1).repeat(1, 1, 1, 1, 1) 

    for i in range(i_max):
        samples = sampler.generate(num_samples=1, y=u0[i].unsqueeze(0))
        result.append(samples)


    visualize = True

    if visualize:
        # TODO: Denormalize results
        gen_sample = result[0][0].permute(1, 2, 0).cpu().detach().numpy()
        gt_sample = u[0].permute(1, 2, 0).cpu().detach().numpy()

        err = gen_sample - gt_sample
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(gen_sample[..., 0])
        axes[0].set_title("Generated")
        axes[0].axis('off')

        axes[1].imshow(gt_sample[..., 0])
        axes[1].set_title("Groundtruth")
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig("output.jpg")