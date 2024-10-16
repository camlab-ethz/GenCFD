import functools

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch import optim

from train import train
import diffusion as dfn_lib
# from model import probabilistic_diffusion as dfn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from utils.callbacks import TqdmProgressBar, TrainStateCheckpoint
from train.train_states import DenoisingModelTrainState
from solvers.sde import EulerMaruyama

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
RNG = torch.Generator(device=device)
RNG.manual_seed(SEED)

RNG_DIFF = torch.Generator()
RNG.manual_seed(SEED)

print(device)


def get_mnist_dataset(split: str, batch_size: int):
    # Define the dataset transformation
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts image to [0, 1]
    ])
    # Load the dataset based on the split
    if split == 'train':
        mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    elif split == 'test':
        mnist_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown split: {split}")

    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=(split == 'train'), num_workers=0)
    return dataloader


########################################### TRAINING ##############################################

# The standard deviation of the normalized dataset.
# This is useful for determining the diffusion scheme and preconditioning
# of the neural network parametrization.
DATA_STD = 0.31

use_model = 'PreconditionedDenoiser'

if use_model == 'PreconditionedDenoiser':
    denoiser_model = dfn_lib.PreconditionedDenoiserUNet(
        out_channels=1,
        rng=RNG,
        num_channels=(64, 128),
        downsample_ratio=(2, 2),
        num_blocks=4,
        noise_embed_dim=128,
        padding_method="circular",
        use_attention=True,
        use_position_encoding=True,
        num_heads=8,
        device=device,
        sigma_data=DATA_STD
    )
elif use_model == 'UNet':
    denoiser_model = dfn_lib.UNet(
        out_channels=1,
        rng=RNG,
        num_channels=(64, 128),
        downsample_ratio=(2, 2),
        num_blocks=4,
        noise_embed_dim=128,
        padding_method="circular",
        use_attention=True,
        use_position_encoding=True,
        num_heads=8,
        device=device
    )
else:
    raise ValueError("Not a valid model")

diffusion_scheme = dfn_lib.Diffusion.create_variance_exploding(
    sigma=dfn_lib.tangent_noise_schedule(),
    data_std=DATA_STD,
)

model = dfn_lib.DenoisingBaseModel(
    # `input_shape` must agree with the expected sample shape (without the batch
    # dimension), which in this case is simply the dimensions of a single MNIST
    # sample.
    input_shape=(1, 28, 28),
    denoiser=denoiser_model,
    noise_sampling=dfn_lib.log_uniform_sampling(
        diffusion_scheme, clip_min=1e-4, uniform_grid=True,
    ),
    noise_weighting=dfn_lib.edm_weighting(data_std=DATA_STD),
    rng=RNG,
    rng_diff=RNG_DIFF,
    device=device
)


cwd = os.getcwd()
workdir = cwd + '/outputs'

if not os.path.exists(workdir):
    raise ValueError("set a propoer workdirectory!")


num_train_steps = 100_0  #@param
train_batch_size = 32  #@param
eval_batch_size = 32  #@param
initial_lr = 0.0  #@param
peak_lr = 1e-4  #@param
warmup_steps = 1000  #@param
end_lr = 1e-6  #@param
ema_decay = 0.999  #@param
ckpt_interval = 1000  #@param
max_ckpt_to_keep = 5  #@param

train_dataloader = get_mnist_dataset('train', train_batch_size)
eval_dataloader = get_mnist_dataset('test', eval_batch_size)
first_batch = next(iter(train_dataloader))
img = first_batch[0].to(device)
labels = first_batch[1].to(device)

noise = torch.randn(img.shape, device=device, generator=RNG)
noised_img = img + noise

# dummy initialization of the Network
model.initialize(img.shape[0])

trainer = train.trainers.DenoisingTrainer(
    model=model,
    optimizer=optim.Adam(model.denoiser.parameters(), lr=peak_lr),
    # We keep track of an exponential moving average of the model parameters
    # over training steps. This alleviates the "color-shift" problems known to
    # exist in the diffusion models.
    ema_decay=ema_decay,
    device=device
)

# TODO: Set to True for a training run!
run_training = False

if run_training:
    train.run(
        train_dataloader=train_dataloader,
        trainer=trainer,
        workdir=workdir,
        total_train_steps=num_train_steps,
        metric_writer=SummaryWriter(log_dir=workdir),
        metric_aggregation_steps=100,
        eval_dataloader=eval_dataloader,
        eval_every_steps=1000,
        num_batches_per_eval=2,
        # callbacks=(Callback(workdir),),
        callbacks=(
            TqdmProgressBar(
                total_train_steps=num_train_steps,
                train_monitors=("train_loss",),
            ),
            TrainStateCheckpoint(
                base_dir=workdir, save_every_n_step=10000
            ),
        )
    )


########################################### INFERENCE #############################################

trained_state = DenoisingModelTrainState.restore_from_checkpoint(
    f"{workdir}/checkpoints/checkpoint_1000.pth", model=model.denoiser, optimizer=trainer.optimizer
)

# Construct the inference function
denoise_fn = trainer.inference_fn_from_state_dict(
    trained_state, use_ema=True, denoiser=denoiser_model, task='superresolver'
)

sampler = dfn_lib.SdeSampler(
    input_shape=(1, 28, 28),
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

samples = sampler.generate(num_samples=4)

########################################### VISUALIZATION #########################################

fig, ax = plt.subplots(1, 4, figsize=(8, 2))
for i in range(4):
  im = ax[i].imshow(samples[i, 0, :, :].cpu().detach().numpy() * 255, cmap="gray", vmin=0, vmax=255)

plt.tight_layout()
plt.savefig("output.jpg")