import sys
import time

import os
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from train import training_loop
from utils.gencfd_utils import (
    get_dataset_loader, 
    get_dataset, 
    create_denoiser,
    create_callbacks
)
from utils.parser_utils import train_args


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 100
RNG = torch.Generator(device=device)
RNG.manual_seed(SEED)

sys.path.append("/usr/local/cuda/bin/ptxas")
# os.environ["WANDB__SERVICE_WAIT"] = "300"

if __name__ == "__main__":

    args = train_args()

    cwd = os.getcwd()
    if args.save_dir is None:
        raise ValueError("Save directory not specified in arguments!")
    savedir = os.path.join(cwd, args.save_dir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
        print(f"Created a directory to store metrics and models: {savedir}")

    train_dataloader, eval_dataloader = get_dataset_loader(
        name=args.dataset, batch_size=args.batch_size, num_worker=args.worker, device=device
    )

    dataset = get_dataset(name=args.dataset, device=device)

    # Get input shape ant output shape and check whether the lead time is included
    if 'lead_time' in dataset.file.variables:
        lead_time, inputs = dataset.__getitem__(0)
        input_shape = inputs.shape
        time_cond = True
    else:
        input_shape = dataset.__getitem__(0).shape
        time_cond = False
    
    out_shape = (dataset.output_channel,) + tuple(input_shape[1:])

    denoising_model = create_denoiser(
        args=args,
        input_shape=input_shape,
        input_channels=dataset.input_channel,
        out_channels=dataset.output_channel,
        rng=RNG,
        device=device
    )

    print(" ")
    print("Denoiser Initialization")

    denoising_model.initialize(batch_size=args.batch_size, time_cond=time_cond)

    # # Print number of Parameters:
    model_params = sum(p.numel() for p in denoising_model.denoiser.parameters() if p.requires_grad)
    print(f"Total number of model parameters: {model_params}")


    trainer = training_loop.trainers.DenoisingTrainer(
        model=denoising_model,
        optimizer=optim.Adam(
            denoising_model.denoiser.parameters(), 
            lr=args.peak_lr),
        ema_decay=args.ema_decay,
        device=device
    )
    # TODO: Implement the following optimizer!
    # torch.optim.AdamW()

    start_train = time.time()

    training_loop.run(
        train_dataloader=train_dataloader,
        trainer=trainer,
        workdir=savedir,
        total_train_steps=args.num_train_steps,
        metric_writer=SummaryWriter(log_dir=savedir),
        metric_aggregation_steps=args.metric_aggregation_steps,
        eval_dataloader=eval_dataloader,
        eval_every_steps=args.eval_every_steps,
        num_batches_per_eval=args.num_batches_per_eval,
        callbacks=create_callbacks(args, savedir)
    )

    end_train = time.time()
    elapsed_train = end_train - start_train
    print(f"Done training. Elapsed time {elapsed_train / 3600} h")