from argparse import ArgumentParser
from typing import Tuple


def add_base_options(parser: ArgumentParser):
    group = parser.add_argument_group('base')
    group.add_argument("--workdir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    

def add_data_options(parser: ArgumentParser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='DataIC_Vel', type=str,
                       # choices=['DataIC_Vel'], # TODO: Implement all possible choices!
                       help="Name of the dataset, available choices")
    group.add_argument("--batch_size", default=5, type=int, help="Choose a batch size")
    group.add_argument("--worker", default=0, type=int,
                       help="Choose the number of worker for parallel processing")


def add_model_options(parser: ArgumentParser):
    group = parser.add_argument_group('model')
    # Model settings
    group.add_argument("--model_type", default='PreconditionedDenoiser', type=str,
                       choices=["PreconditionedDenoiser", "UNet"], # TODO: ADD further models!
                       help="Choose a valid Neural Network Model architecture")
    group.add_argument("--num_channels", default=(64, 128), type=Tuple[int, ...], 
                       help="Number of channels for down and upsampling")
    group.add_argument("--downsampling_ratio", default=(2, 2), type=Tuple[int, ...],
                       help="Choose a downsampling ratio")
    # Attention settings
    group.add_argument("--use_attention", default=True, type=bool, 
                       help="Choose if attention blocks should be used")
    group.add_argument("--num_blocks", default=4, type=int, 
                       help="Choose number of Attention blocks")
    group.add_argument("--num_heads", default=8, type=int, 
                       help="Choose number of heads for multihead attention")
    group.add_argument("--normalize_qk", default=False, type=bool,
                       help="Choose if Query and Key matrix should be normalized")
    # Embedding settings
    group.add_argument("--noise_embed_dim", default=128, type=int, 
                       help="Choose noise embedding dimension")
    group.add_argument("--use_position_encoding", default=True, type=bool,
                       help="Use position encoding True or False")
    # General settings
    group.add_argument("--padding_method", default="circular", type=str,
                       choices=["circular", "constant", "lonlat", "latlon"],
                       help="Choose a proper padding method from the list of choices")
    group.add_argument("--dropout_rate", default=0.0, type=float,
                       help="Choose a proper dropout rate")
    group.add_argument("--use_hr_residual", default=False, type=bool,
                       help="Dropout rate for classifier-free guidance")
    group.add_argument("--sigma_data", default=0.5, type=float,
                       help="This can be a fixed in [0, 1] or learnable parameter")
    

def add_denoiser_options(parser: ArgumentParser):
    group = parser.add_argument_group('denoiser')
    group.add_argument('--noise_sampling', 
                       default='log_uniform_sampling', 
                       type=str,
                       choices=['log_uniform_sampling', 
                                'time_uniform_sampling',
                                'normal_sampling'],
                       help="Choose a valid noise sampler from the list of choices")
    group.add_argument('--noise_weighting', default='edm_weighting', type=str,
                       choices=['edm_weighting', 'likelihood_weighting'],
                       help='Choose a valid weighting method from the list of choices')
    group.add_argument('--consistent_weight', default=0.0, type=float, 
                       help='weighting of some loss terms')
    

def add_trainer_options(parser: ArgumentParser):
    group = parser.add_argument_group('trainer')
    # EMA ... Exponential Moving Average
    group.add_argument('--ema_decay', default=0.999, type=float, 
                       help='Choose a decay rate for the EMA model parameters')
    group.add_argument('--peak_lr', default=1e-4, type=str,
                       help="Choose a learning rate for the Adam optimizer")
    group.add_argument('--task', default='solver', type=str, 
                       choices=['solver', 'superresolver'],
                       help='Decide whether the model should be used as a solver or superresolver')
    

def add_training_options(parser: ArgumentParser):
    group = parser.add_argument_group('training')
    group.add_argument('--num_train_steps', default=10_000, type=int,
                       help='Choose number of training steps')
    group.add_argument('--Metric_aggregation_steps', default=100, type=int,
                       help='trainer runs this number of steps until training metrics are aggregated')
    group.add_argument('--eval_every_steps', default=100, type=int, 
                       help='Period at which an evaluation loop runs')
    group.add_argument('--num_batches_per_eval', default=2, type=int,
                       help='Number of steps until evaluation metrics are aggregated')
    group.add_argument('--run_sanity_eval_batch', default=True, type=bool,
                       help='Sanity check to spot early mistakes or runtime issues')
    group.add_argument('--save_load_checkpoints', default=True, type=bool,
                       help="Saves or Loads parameters from a checkpoint")
    group.add_argument('--save_every_n_steps', default=5000, type=int,
                       help="Saves a checkpoint of the model and optimizer after every n steps")

def train_args():
    parser = ArgumentParser()
    add_data_options(parser)
    add_model_options(parser)
    add_denoiser_options(parser)
    add_trainer_options(parser)
    add_training_options(parser)
    return parser.parse_args()