import sys
import time

import os
import torch
from torch import optim
# from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from train import training_loop
import diffusion as dfn_lib
from utils.callbacks import TqdmProgressBar, TrainStateCheckpoint
from dataloader.dataset import get_dataset_loader, get_dataset
from utils.parser_utils import train_args


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
RNG = torch.Generator(device=device)
RNG.manual_seed(SEED)

sys.path.append("/usr/local/cuda/bin/ptxas")

# os.environ["WANDB__SERVICE_WAIT"] = "300"

if __name__ == "__main__":

    args = train_args()

    # Dataloader parameters set manually
    generate_during_eval = False
    ckpt_interval = 100
    max_ckpt_to_keep = 3
    log_train_every_steps = 500
    if generate_during_eval:
        eval_batch_size = 5
        eval_every_steps = log_train_every_steps * 8
        num_batches_per_eval = min(100 // eval_batch_size, 4)
    else:
        eval_batch_size = 5
        eval_every_steps = log_train_every_steps * 2
        num_batches_per_eval = 100 // eval_batch_size


    # dataset = DataIC_Vel()

    # train_size = int(0.8 * len(dataset))  # 80% for training
    # eval_size = len(dataset) - train_size  # 20% for evaluation 

    # train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
    # eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=train_batch_size, shuffle=True)

    train_dataloader, eval_dataloader = get_dataset_loader(
        name=args.dataset, batch_size=args.batch_size, num_worker=args.worker
    )

    train_dataset = get_dataset(name=args.dataset)

    breakpoint()

    # Model parameters set manually
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


    inp_shape = train_dataset.__getitem__(0).shape
    out_shape = (out_channels,) + tuple(inp_shape[1:])

    # Workdirectory set manually
    cwd = os.getcwd()
    workdir = cwd + '/outputs'

    if not os.path.exists(workdir):
        raise ValueError("set a propoer workdirectory!")

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

    # diffusion_scheme = dfn_lib.Diffusion.create_variance_exploding(
    #     sigma=dfn_lib.tangent_noise_schedule(),
    #     data_std=DATA_STD,
    # )

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
    model.initialize(batch_size=args.batch_size)

    # Print number of Parameters:
    model_params = sum(p.numel() for p in model.denoiser.parameters() if p.requires_grad)
    print(f"Total number of model parameters: {model_params}")


    trainer = training_loop.trainers.DenoisingTrainer(
        model=model,
        optimizer=optim.Adam(model.denoiser.parameters(), lr=peak_lr),
        ema_decay=ema_decay,
        device=device
    )
    torch.optim.AdamW()
    start_train = time.time()

    training_loop.run(
        train_dataloader=train_dataloader,
        trainer=trainer,
        workdir=workdir,
        total_train_steps=num_train_steps,
        metric_writer=SummaryWriter(log_dir=workdir),
        metric_aggregation_steps=100,
        eval_dataloader=eval_dataloader,
        eval_every_steps=100,
        num_batches_per_eval=2,
        # callbacks=(Callback(workdir),),
        callbacks=(
            TqdmProgressBar(
                total_train_steps=num_train_steps,
                train_monitors=("train_loss",),
            ),
            TrainStateCheckpoint(
                base_dir=workdir, save_every_n_step=5000
            ),
        )
    )
    end_train = time.time()
    elapsed_train = end_train - start_train
    print(f"Done training. Elapsed time {elapsed_train / 3600} h")