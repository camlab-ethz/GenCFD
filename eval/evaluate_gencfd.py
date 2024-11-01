""""
Main file to run evaluation for the gencfd model
"""
import sys
import random
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
    replace_args
)
from eval.metrics.stats_recorder import StatsRecorder
from eval import evaluation_loop


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
RNG = torch.Generator(device=device)
RNG.manual_seed(SEED)

# sys.path.append("/usr/local/cuda/bin/ptxas")

if __name__=="__main__":

    print(f'Used device {device}')
    
    # get arguments for inference
    args = inference_args()

    cwd = os.getcwd()
    if args.model_dir is None:
        raise ValueError("Path to a trained model is not specified!")
    model_dir = os.path.join(cwd, args.model_dir, "checkpoints")
    if not os.path.exists(model_dir):
        raise ValueError("Wrong Path it doesn't exist!")
    
    # read configurations which were used to train the model
    train_args = load_json_file(
        os.path.join(cwd, args.model_dir, "training_config.json")
    )

    # determine whether the json file exists
    if train_args is None:
        # json file does not exist and train_args is None
        dataset, time_cond = get_dataset(
            name=args.dataset, device=device, is_time_dependent=True
        )
        if 'lead_time' in dataset.file.variables:
            lead_time, inputs = dataset.__getitem__(0)
            input_shape = inputs.shape
        else:
            input_shape = dataset.__getitem__(0).shape
    
        out_shape = (dataset.output_channel,) + tuple(input_shape[1:])
    else:
        # json file exists, use parameters which were used during training
        dataset = get_dataset(name=args.dataset, device=device)

        if SEED != train_args["seed"]:
            # to get the same training data if the same dataset is used
            RNG.manual_seed(train_args["seed"]) 

        # check if the correct device is currently in use
        if str(device) != train_args["device"]:
            raise ValueError(f"Wrong device, device needs to be {train_args['device']}")

        # replace every argument from train_args besides the dataset name!
        replace_args(args, train_args) 

        input_shape = tuple(train_args['input_shape'])
        out_shape = tuple(train_args['out_shape'])
        time_cond = train_args['time_cond']

    # get the dataloader
    dataloader = get_dataset_loader(
        name=args.dataset, 
        batch_size=args.batch_size, 
        num_worker=args.worker, 
        split=False,
        device=device
    )

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

    trainer = DenoisingTrainer(
        model=denoising_model,
        optimizer=optim.Adam(
            denoising_model.denoiser.parameters(), 
            lr=args.peak_lr),
        ema_decay=args.ema_decay,
        device=device
    )

    print(" ")
    print("Load Model Parameters")

    latest_model_path = get_latest_checkpoint(model_dir)

    trained_state = DenoisingModelTrainState.restore_from_checkpoint(
        latest_model_path, model=denoising_model.denoiser, optimizer=trainer.optimizer
    )

    # Construct the inference function
    denoise_fn = trainer.inference_fn_from_state_dict(
        trained_state, 
        use_ema=True, 
        denoiser=denoising_model.denoiser, 
        task=args.task,
        lead_time=time_cond
    )

    sampler = create_sampler(
        args=args,
        input_shape=out_shape, 
        denoise_fn=denoise_fn,
        rng=RNG, 
        device=device
    )

    # initialize stats_recorder to keep track of metrics
    stats_recorder = StatsRecorder(
        batch_size=args.batch_size, 
        ndim=len(out_shape)-1, 
        channels=dataset.output_channel, 
        data_shape=out_shape,
        device=device
    )

    print(" ")
    print(f"Run Evaluation Loop with {args.monte_carlo_samples} Monte Carlo Samples")
    start_train = time.time()

    evaluation_loop.run(
        sampler=sampler,
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

    # Uncomment the following line for tracking memory!
    # print(torch.cuda.memory_summary())

    end_train = time.time()
    elapsed_train = end_train - start_train
    print(" ")
    print(f"Done evaluation. Elapsed time {elapsed_train / 3600} h")