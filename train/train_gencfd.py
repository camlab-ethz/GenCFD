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
# Set the cache size and debugging for torch.compile before importing torch
# os.environ["TORCH_LOGS"] = "all"  # or any of the valid log settings
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from train import training_loop
from utils.gencfd_utils import (
    get_dataset_loader,
    get_buffer_dict,
    create_denoiser,
    create_callbacks,
    save_json_file
)
from utils.parser_utils import train_args

torch.set_float32_matmul_precision('high') # Better performance on newer GPUs!
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0

# Setting global seed for reproducibility
torch.manual_seed(SEED)  # For CPU operations
torch.cuda.manual_seed(SEED)  # For GPU operations
torch.cuda.manual_seed_all(SEED)  # Ensure all GPUs (if multi-GPU) are set

if __name__ == "__main__":
    print(" ")
    print(f'Used device {device}')

    # get arguments for training
    args = train_args()

    cwd = os.getcwd()
    if args.save_dir is None:
        raise ValueError("Save directory not specified in arguments!")
    savedir = os.path.join(cwd, args.save_dir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
        print(f"Created a directory to store metrics and models: {savedir}")

    split_ratio = 0.8
    train_dataloader, eval_dataloader, dataset, time_cond = get_dataset_loader(
        name=args.dataset, 
        batch_size=args.batch_size, 
        num_worker=args.worker, 
        split=True,
        split_ratio=split_ratio
    )

    # Create a buffer dictionary to store normalization parameters in the NN
    buffer_dict = get_buffer_dict(dataset=dataset, device=device)

    # Save parameters in a JSON File
    save_json_file(
        args=args, 
        time_cond=time_cond, 
        split_ratio=split_ratio,
        out_shape=dataset.output_shape, # output shape of the prediction 
        input_channel=dataset.input_channel,
        output_channel=dataset.output_channel,
        spatial_resolution=dataset.spatial_resolution,
        device=device, 
        seed=SEED
    )

    denoising_model = create_denoiser(
        args=args,
        input_channels=dataset.input_channel,
        out_channels=dataset.output_channel,
        spatial_resolution=dataset.spatial_resolution,
        time_cond=time_cond,
        device=device,
        dtype=args.dtype,
        buffer_dict=buffer_dict
    )
    
    with torch.no_grad():
        # Warmup round to check if model is running as intended
        denoising_model.initialize(batch_size=args.batch_size, time_cond=time_cond)

    # Print number of Parameters:
    model_params = sum(p.numel() for p in denoising_model.denoiser.parameters() if p.requires_grad)
    print(" ")
    print(f"Total number of model parameters: {model_params}")
    print(" ")

    trainer = training_loop.trainers.DenoisingTrainer(
        model=denoising_model,
        optimizer=optim.AdamW(
            denoising_model.denoiser.parameters(), 
            lr=args.peak_lr,
            weight_decay=args.weight_decay),    
        device=device,
        ema_decay=args.ema_decay,
        store_ema=False, # Changed manually
        track_memory=args.track_memory,
        use_mixed_precision=args.use_mixed_precision,
        is_compiled=args.compile
    )

    start_train = time.time()

    training_loop.run(
        train_dataloader=train_dataloader,
        trainer=trainer,
        workdir=savedir,
        total_train_steps=args.num_train_steps,
        metric_writer=SummaryWriter(log_dir=savedir),
        metric_aggregation_steps=args.metric_aggregation_steps,
        # Only do evaluation while training if the model is not compiled
        eval_dataloader=eval_dataloader if not args.compile else None, 
        eval_every_steps=args.eval_every_steps,
        num_batches_per_eval=args.num_batches_per_eval,
        callbacks=create_callbacks(args, savedir),
        compile_model=args.compile
    )

    end_train = time.time()
    elapsed_train = end_train - start_train
    print(f"Done training. Elapsed time {elapsed_train / 3600} h")