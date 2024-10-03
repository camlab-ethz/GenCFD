import functools

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch import optim

from train import train
import diffusion as dfn_lib
# from model import probabilistic_diffusion as dfn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from utils.callbacks import Callback, TqdmProgressBar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
RNG = torch.Generator(device=device)
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

    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=(split == 'train'), num_workers=2)
    return dataloader


# The standard deviation of the normalized dataset.
# This is useful for determining the diffusion scheme and preconditioning
# of the neural network parametrization.
DATA_STD = 0.31

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

diffusion_scheme = dfn_lib.Diffusion.create_variance_exploding(
    sigma=dfn_lib.tangent_noise_schedule(),
    data_std=DATA_STD,
)

model = dfn_lib.DenoisingModel(
    # `input_shape` must agree with the expected sample shape (without the batch
    # dimension), which in this case is simply the dimensions of a single MNIST
    # sample.
    input_shape=(28, 28, 1),
    denoiser=denoiser_model,
    noise_sampling=dfn_lib.log_uniform_sampling(
        diffusion_scheme, clip_min=1e-4, uniform_grid=True,
    ),
    noise_weighting=dfn_lib.edm_weighting(data_std=DATA_STD),
    rng=RNG,
    seed=SEED,
    device=device
)

num_train_steps = 100_000  #@param
workdir = "/tmp/diffusion_demo_mnist"  #@param
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
test_dataloader = get_mnist_dataset('test', eval_batch_size)
first_batch = next(iter(train_dataloader))
img = first_batch[0].to(device)
labels = first_batch[1].to(device)

noise = torch.randn(img.shape, device=device, generator=RNG)
noised_img = img + noise

# Initialize model by running a dummy forward pass!
dummy_input = torch.randn(
    img.shape, dtype=torch.float32, device=device, generator=RNG
    )
dummy_sigma = torch.randn(
    labels.shape, dtype=torch.float32, device=device, generator=RNG
    )
# model.to(device) # TODO: model should have an input to device!!!
_ = model.denoiser(dummy_input, dummy_sigma)

# breakpoint()

trainer = train.trainers.DenoisingTrainer(
    model=model,
    optimizer=optim.Adam(model.denoiser.parameters(), lr=peak_lr),
    # We keep track of an exponential moving average of the model parameters
    # over training steps. This alleviates the "color-shift" problems known to
    # exist in the diffusion models.
    ema_decay=ema_decay,
    device=device
)


train.run(
    train_dataloader=train_dataloader,
    trainer=trainer,
    workdir=workdir,
    total_train_steps=num_train_steps,
    metric_writer=SummaryWriter(log_dir=workdir),
    metric_aggregation_steps=100,
    eval_dataloader=None,
    eval_every_steps=1000,
    num_batches_per_eval=2,
    # callbacks=(Callback(workdir),),
    callbacks=(
        TqdmProgressBar(
            total_train_steps=num_train_steps,
            train_monitors=("train_loss",),
        ),
    )
)