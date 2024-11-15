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

"""Main File to run Training for GenCFD."""

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
    create_callbacks,
    save_json_file
)
from utils.parser_utils import train_args


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
RNG = torch.Generator(device=device)
RNG.manual_seed(SEED)

if __name__ == "__main__":

    args = train_args()

    cwd = os.getcwd()
    if args.save_dir is None:
        raise ValueError("Save directory not specified in arguments!")
    savedir = os.path.join(cwd, args.save_dir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
        print(f"Created a directory to store metrics and models: {savedir}")

    split_ratio = 0.8
    train_dataloader, eval_dataloader = get_dataset_loader(
        name=args.dataset, 
        batch_size=args.batch_size, 
        num_worker=args.worker, 
        split=True,
        split_ratio=split_ratio,
        device=device
    )

    dataset, time_cond = get_dataset(name=args.dataset, device=device, is_time_dependent=True)

    # Get input shape ant output shape and check whether the lead time is included
    if time_cond:
        batch = dataset.__getitem__(0)
        lead_time, inputs = batch['lead_time'], batch['data']
        input_shape = inputs.shape
    else:
        input_shape = dataset.__getitem__(0).shape
    
    out_shape = (dataset.output_channel,) + tuple(input_shape[1:])

    # extract normalization values from the used dataset, these can also be computed if not provided!
    # these values are then stored inside the NN model in buffers, so they don't get updated
    mean_training_input = torch.tensor(dataset.mean_training_input, dtype=args.dtype, device=device)
    mean_training_output = torch.tensor(dataset.mean_training_output, dtype=args.dtype, device=device)
    std_training_input = torch.tensor(dataset.std_training_input, dtype=args.dtype, device=device)
    std_training_output = torch.tensor(dataset.std_training_output, dtype=args.dtype, device=device)

    buffer_dict = {
        'mean_training_input': mean_training_input,
        'mean_training_output': mean_training_output,
        'std_training_input': std_training_input,
        'std_training_output': std_training_output
    }

    # Save parameters in a JSON File
    save_json_file(
        args=args, 
        time_cond=time_cond, 
        split_ratio=split_ratio,
        input_shape=input_shape,
        out_shape=out_shape, 
        input_channel=dataset.input_channel,
        output_channel=dataset.output_channel,
        device=device, 
        seed=SEED
    )

    print(" ")
    print("Denoiser Initialization")

    denoising_model = create_denoiser(
        args=args,
        input_shape=input_shape,
        input_channels=dataset.input_channel,
        out_channels=dataset.output_channel,
        rng=RNG,
        device=device,
        dtype=args.dtype,
        buffer_dict=buffer_dict
    )

    denoising_model.initialize(batch_size=args.batch_size, time_cond=time_cond)

    # Print number of Parameters:
    model_params = sum(p.numel() for p in denoising_model.denoiser.parameters() if p.requires_grad)
    print(f"Total number of model parameters: {model_params}")

    trainer = training_loop.trainers.DenoisingTrainer(
        model=denoising_model,
        optimizer=optim.AdamW(
            denoising_model.denoiser.parameters(), 
            lr=args.peak_lr,
            weight_decay=args.weight_decay),    
        device=device,
        ema_decay=args.ema_decay,
        track_memory=args.track_memory,
        use_mixed_precision=args.use_mixed_precision
    )

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