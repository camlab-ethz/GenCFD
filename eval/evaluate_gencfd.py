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

"""Main File to Run Inference.

Options are to compute statistical metrics or visualize results.
"""

import time
import os
import matplotlib.pyplot as plt
import torch
from torch import optim

from train.train_states import DenoisingModelTrainState
from train.trainers import DenoisingTrainer
from utils.parser_utils import inference_args
from utils.gencfd_utils import (
    get_dataset_loader, 
    get_dataset, 
    create_denoiser,
    create_sampler,
    get_latest_checkpoint,
    load_json_file,
    replace_args,
    get_buffer_dict
)
from eval.metrics.stats_recorder import StatsRecorder
from eval import evaluation_loop


torch.set_float32_matmul_precision('high') # Better performance on newer GPUs!
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0

# Setting global seed for reproducibility
torch.manual_seed(SEED)  # For CPU operations
torch.cuda.manual_seed(SEED)  # For GPU operations
torch.cuda.manual_seed_all(SEED)  # Ensure all GPUs (if multi-GPU) are set

if __name__=="__main__":

    print(f'Used device {device}')
    
    # get arguments for inference
    args = inference_args()

    cwd = os.getcwd()
    if args.model_dir is None:
        raise ValueError("Path to a trained model is not specified!")
    model_dir = os.path.join(cwd, args.model_dir, "checkpoints")
    if not os.path.exists(model_dir):
        raise ValueError(f"Wrong Path, {args.model_dir} doesn't exist!")
    
    # read configurations which were used to train the model
    train_args = load_json_file(
        os.path.join(cwd, args.model_dir, "training_config.json")
    )

    dataloader, dataset, time_cond = get_dataset_loader(
        name=args.dataset, 
        batch_size=args.batch_size, 
        num_worker=args.worker, 
        prefetch_factor=2, # Default DataLoader Value
        split=False
    )

    input_shape = dataset.input_shape
    out_shape = dataset.output_shape
    spatial_resolution = dataset.spatial_resolution

    if train_args:
        # check if the correct device is currently in use
        if str(device) != train_args["device"]:
            raise ValueError(f"Wrong device, device needs to be {train_args['device']}")

        # replace every argument from train_args besides the dataset name!
        replace_args(args, train_args) 

        # Check if the arguments used for training are the same as the evaluation dataset
        # assert spatial_resolution == tuple(train_args['spatial_resolution']), \
        #     f"spatial_resolution should be {tuple(train_args['input_shape'])} " \ 
        #     f"and not {spatial_resolution}"
        assert spatial_resolution == tuple(train_args['spatial_resolution']), \
            f"spatial_resolution should be {tuple(train_args['spatial_resolution'])} " \
            f"and not {spatial_resolution}"
        assert out_shape == tuple(train_args['out_shape']), \
            f"out_shape should be {tuple(train_args['input_shape'])} and not {out_shape}"
        # assert time_cond == train_args['time_cond'], \
        #     f"time_cond should be {train_args['time_cond']} and not {time_cond}"
    
    # Dummy buffer values, for initialization! Necessary to load the model parameters
    buffer_dict = get_buffer_dict(dataset=dataset, create_dummy=True)

    # the compute_dtype needs to be the same as used for the trained model!
    denoising_model = create_denoiser(
        args=args,
        input_channels=dataset.input_channel,
        out_channels=dataset.output_channel,
        spatial_resolution=spatial_resolution,
        time_cond=time_cond,
        device=device,
        dtype=args.dtype,
        buffer_dict=buffer_dict
    )

    with torch.no_grad():
        denoising_model.initialize(batch_size=args.batch_size, time_cond=time_cond)

    # Print number of Parameters:
    model_params = sum(p.numel() for p in denoising_model.denoiser.parameters() if p.requires_grad)
    print(" ")
    print(f"Total number of model parameters: {model_params}")
    print(" ")

    # Rebuild the trainer used for training
    trainer = DenoisingTrainer(
        model=denoising_model,
        optimizer=optim.AdamW(
            denoising_model.denoiser.parameters(), 
            lr=args.peak_lr,
            weight_decay=args.weight_decay),
        device=device,
        ema_decay=args.ema_decay,
        store_ema=False, 
        track_memory=False,
        use_mixed_precision=args.use_mixed_precision,
        is_compiled=args.compile
    )

    print("Load Model Parameters")
    print(" ")

    latest_model_path = get_latest_checkpoint(model_dir)

    trained_state = DenoisingModelTrainState.restore_from_checkpoint(
        latest_model_path, model=denoising_model.denoiser, optimizer=trainer.optimizer
    )

    # Retrieve the normalization buffer (mean and std tensors)
    buffers = dict(denoising_model.denoiser.named_buffers())
    for key, tensor in buffers.items():
        if tensor is not None:  # put tensor on same device!
            buffers[key] = tensor.to(device=device)

    # Construct the inference function
    denoise_fn = trainer.inference_fn_from_state_dict(
        trained_state, 
        use_ema=False, # Changed! 
        denoiser=denoising_model.denoiser, 
        task=args.task,
        lead_time=time_cond
    )

    # Create Sampler
    sampler = create_sampler(
        args=args,
        input_shape=out_shape, 
        denoise_fn=denoise_fn,
        device=device
    )

    # Initialize stats_recorder to keep track of metrics
    stats_recorder = StatsRecorder(
        batch_size=args.batch_size, 
        ndim=len(out_shape)-1, 
        channels=dataset.output_channel, 
        data_shape=out_shape,
        monte_carlo_samples=args.monte_carlo_samples,
        num_samples=1000, # Choose 1000 random pixel values
        device=device
    )

    if args.compute_metrics:
        print(f"Run Evaluation Loop with {args.monte_carlo_samples} Monte Carlo Samples and Batch Size {args.batch_size}")
    if args.visualize:
        print(f"Run Visualization Loop")
        
    start_train = time.time()

    evaluation_loop.run(
        sampler=sampler,
        buffers=buffers,
        monte_carlo_samples=args.monte_carlo_samples,
        stats_recorder=stats_recorder,
        dataloader=dataloader,
        dataset=dataset,
        dataset_module=args.dataset,
        time_cond=time_cond,
        compute_metrics=args.compute_metrics,
        visualize=args.visualize,
        device=device,
        save_dir=args.save_dir
    )

    end_train = time.time()
    elapsed_train = end_train - start_train
    print(" ")
    print(f"Done evaluation. Elapsed time {elapsed_train / 3600} h")