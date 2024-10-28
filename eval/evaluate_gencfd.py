import sys
import random

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
RNG = torch.Generator(device=device)
RNG.manual_seed(SEED)

# sys.path.append("/usr/local/cuda/bin/ptxas")

if __name__=="__main__":

    print(f'Used device {device}')
    
    args = inference_args()

    cwd = os.getcwd()
    if args.model_dir is None:
        raise ValueError("Path to a trained model is not specified!")
    model_dir = os.path.join(cwd, args.model_dir, "checkpoints")
    if not os.path.exists(model_dir):
        raise ValueError("Wrong Path it doesn't exist!")
    
    train_args = load_json_file(
        os.path.join(cwd, args.model_dir, "training_config.json")
    )

    if SEED != train_args["seed"]:
        # to get the same training data
        RNG.manual_seed(train_args["seed"]) 

    # check if the correct device is currently in use
    if str(device) != train_args["device"]:
        raise ValueError(f"Wrong device, device needs to be {train_args['device']}")

    replace_args(args, train_args)

    train_dataloader, eval_dataloader = get_dataset_loader(
        name=args.dataset, 
        batch_size=args.batch_size, 
        num_worker=args.worker, 
        split=True,
        split_ratio=train_args['split_ratio'],
        device=device
    )

    # dataset = get_dataset(name=train_args['dataset'], device=device)

    denoising_model = create_denoiser(
        args=args,
        input_shape=tuple(train_args['input_shape']),
        input_channels=train_args['input_channel'],
        out_channels=train_args['output_channel'],
        rng=RNG,
        device=device
    )

    print(" ")
    print("Denoiser Initialization")

    denoising_model.initialize(batch_size=args.batch_size, time_cond=train_args['time_cond'])

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
        lead_time=train_args['time_cond']
    )

    print(" ")
    print("Sampler Initialization")

    sampler = create_sampler(
        args=args,
        input_shape=tuple(train_args['out_shape']), 
        denoise_fn=denoise_fn,
        rng=RNG, 
        device=device
    )

    # rand_idx = random.randint(0, len(dataset))

    # generate = functools.partial(sampler.generate, num_samples=1)
    if train_args['time_cond']:
        time, normalized_data_batch = next(iter(eval_dataloader))
    else:
        normalized_data_batch = next(iter(eval_dataloader))
        time=None

    u0 = normalized_data_batch[0, :train_args['output_channel'], ...].unsqueeze(0)
    u = normalized_data_batch[0, train_args['output_channel']:, ...].unsqueeze(0)

    # Only necessary if num_samples > 1 uncomment the following line
    # u0_norm_rep = u0.unsqueeze(1).repeat(1, 1, 1, 1, 1) 

    # for i in range(i_max):
    #     samples = sampler.generate(num_samples=1, y=u0[i].unsqueeze(0))
    #     result.append(samples)

    samples = sampler.generate(num_samples=1, y=u0, lead_time=time[0] if time is not None else time)

    # TODO: for tracking memory!
    # print(torch.cuda.memory_summary())

    visualize = False

    # if visualize:
        # TODO: Denormalize results
        # import pyvista as pv
        # import numpy as np

        # gen_sample = samples[0].permute(1, 2, 3, 0).cpu().detach().numpy()
        # gt_sample = u[0].permute(1, 2, 3, 0).cpu().detach().numpy()

        # np.savez('solutions.npz', gen_sample=gen_sample, gt_sample=gt_sample)

        # volume = pv.wrap(gen_sample[..., 0])
        # plotter = pv.Plotter(off_screen=True)
        # plotter.add_volume(volume, opacity="sigmoid", cmap="viridis", shade=True)
        # plotter.screenshot("gen_3d_image.png")
        

    if visualize:
        # TODO: Denormalize results
        gen_sample = samples[0].permute(1, 2, 0).cpu().detach().numpy()
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