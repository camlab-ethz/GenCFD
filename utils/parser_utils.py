from argparse import ArgumentParser
from typing import Tuple
import torch

def parse_tuple(value):
    if value is None or value.lower() == "none":
        return None
    try:
        return tuple(map(int, value.strip('()').split(',')))
    except ValueError:
        raise ValueError(f"Invalid tuple format: {value}")


def add_base_options(parser: ArgumentParser):
    """General base arguments for training and inference"""

    group = parser.add_argument_group('base')
    group.add_argument("--work_dir", default="datasets", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    group.add_argument('--save_dir', default=None, type=str,
                       help='Specify your directory where training or evaluation results should be saved')
    group.add_argument('--model_dir', default=None, type=str,
                       help='Set a path to a pretrained model for inference')
    group.add_argument('--unconditional', default=False, type=bool,
                       help='Run unconditional training and inference if set to True')
    group.add_argument('--dtype', default=torch.float32, type=torch.dtype,
                       help="Set the precision for PyTorch tensors by defining the dtype")
    

def add_data_options(parser: ArgumentParser):
    """Relevant parser arguments for the dataloader"""

    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='DataIC_Vel', type=str,
                       choices=[
                           'DataIC_Vel', 'DataIC_3D_Time', 'DataIC_3D_Time_TG',
                           'ConditionalDataIC_Vel', 'ConditionalDataIC_3D'
                        ],
                       help="Name of the dataset, available choices")
    group.add_argument("--batch_size", default=5, type=int, help="Choose a batch size")
    group.add_argument("--worker", default=0, type=int,
                       help="Choose the number of worker for parallel processing")


def add_model_options(parser: ArgumentParser):
    """Relevant parser arguments for the UNet architecture"""

    group = parser.add_argument_group('model')
    # Model settings
    group.add_argument("--model_type", default='PreconditionedDenoiser', type=str,
                       choices=[
                           "PreconditionedDenoiser", "UNet",
                           "PreconditionedDenoiser3D", "UNet3D"
                        ], 
                       help="Choose a valid Neural Network Model architecture")
    group.add_argument("--num_channels", default=(64, 128), type=parse_tuple, 
                       help="Number of channels for down and upsampling")
    group.add_argument("--downsample_ratio", default=(2, 2), type=parse_tuple,
                       help="Choose a downsample ratio")
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
    group.add_argument("--resize_to_shape", default=None, type=parse_tuple,
                       help="Choose a shape to resize inside the UNet")
    

def add_denoiser_options(parser: ArgumentParser):
    """Relevant parameter for the DenoisingModel"""

    group = parser.add_argument_group('denoiser')
    group.add_argument('--diffusion_scheme', 
                       default='create_variance_exploding',
                       choices=[
                           'create_variance_preserving',
                           'create_variance_exploding'   
                       ],
                       help='Choose a valid diffusion scheme')
    group.add_argument('--sigma',
                       default='exponential_noise_schedule',
                       choices=[
                           'exponential_noise_schedule',
                           'power_noise_schedule',
                           'tangent_noise_schedule'
                       ],
                       help='Choose a valid noise scheduler sigma')
    group.add_argument('--noise_sampling', 
                       default='log_uniform_sampling', 
                       type=str,
                       choices=[
                            'log_uniform_sampling', 
                            'time_uniform_sampling',
                            'normal_sampling'
                        ],
                       help="Choose a valid noise sampler from the list of choices")
    group.add_argument('--noise_weighting', default='edm_weighting', type=str,
                       choices=['edm_weighting', 'likelihood_weighting'],
                       help='Choose a valid weighting method from the list of choices')
    group.add_argument('--num_eval_noise_levels', default=5, type=int,
                       help='Set number of noise levels for evaluation during training')
    group.add_argument('--num_eval_cases_per_lvl', default=1, type=int,
                       help='Set number of evaluation samples per noise level')
    group.add_argument('--min_eval_noise_lvl', default=1e-3, type=float,
                       help='Minimum noise level during evaluation')
    group.add_argument('--max_eval_noise_lvl', default=50.0, type=float,
                       help='Maximum noise level during evaluation')
    group.add_argument('--consistent_weight', default=0.0, type=float,
                       help='Set weighting for some loss terms')
    

def add_trainer_options(parser: ArgumentParser):
    """Parser Arguments for the Trainer"""

    group = parser.add_argument_group('trainer')
    # EMA ... Exponential Moving Average
    group.add_argument('--ema_decay', default=0.999, type=float, 
                       help='Choose a decay rate for the EMA model parameters')
    group.add_argument('--peak_lr', default=1e-4, type=float,
                       help="Choose a learning rate for the Adam optimizer")
    group.add_argument('--task', default='solver', type=str, 
                       choices=['solver', 'superresolver'],
                       help='Decide whether the model should be used as a solver or superresolver')
    

def add_training_options(parser: ArgumentParser):
    """Parser arguments for the training loop"""

    group = parser.add_argument_group('training')
    group.add_argument('--num_train_steps', default=10_000, type=int,
                       help='Choose number of training steps')
    group.add_argument('--metric_aggregation_steps', default=500, type=int,
                       help='trainer runs this number of steps until training metrics are aggregated')
    group.add_argument('--eval_every_steps', default=1000, type=int, 
                       help='Period at which an evaluation loop runs')
    group.add_argument('--num_batches_per_eval', default=2, type=int,
                       help='Number of steps until evaluation metrics are aggregated')
    group.add_argument('--run_sanity_eval_batch', default=True, type=bool,
                       help='Sanity check to spot early mistakes or runtime issues')
    group.add_argument('--checkpoints', default=True, type=bool,
                       help="Saves or Loads parameters from a checkpoint")
    group.add_argument('--save_every_n_steps', default=5000, type=int,
                       help="Saves a checkpoint of the model and optimizer after every n steps")
    
def add_sampler_options(parser: ArgumentParser):
    """Parser arguments for the sampler"""

    group = parser.add_argument_group('sampler')
    group.add_argument('--time_step_scheduler', default='edm_noise_decay', type=str,
                       choices=[
                           'edm_noise_decay',
                           'exponential_noise_decay',
                           'uniform_time'
                       ],
                       help='Choose a valid time step scheduler for solving an SDE')
    group.add_argument('--sampling_steps', default=128, type=int,
                       help='Define sampling steps for solving the SDE, min value should be 32')
    group.add_argument('--apply_denoise_at_end', default=True, type=bool,
                       help='If True applies the denoise function another time to the terminal states')
    group.add_argument('--return_full_paths', default=False, type=bool,
                       help='If True the output of .generate() and .denoise() will contain the complete sampling path')
    group.add_argument('--rho', default=7, type=int,
                       help='Set decay rate for the noise over time')
    

def add_sde_options(parser: ArgumentParser):
    """Parser arguments for the Euler Maruyama Method"""

    group = parser.add_argument_group('sde')
    group.add_argument('--integrator', default='EulerMaruyame', type=str,
                       help='Choose a valid SDE Solver')
    group.add_argument('--time_axis_pos', default=0, type=int,
                       help='Defines the index where the time axis should be placed')
    group.add_argument('--terminal_only', default=True, type=bool,
                       help='If set to False returns the full path otherwise only the terminal state')
    
def add_evaluation_options(parser: ArgumentParser):
    """Parser arguments to compute Metrics for the inference pipeline"""
    group = parser.add_argument_group('evaluation')
    group.add_argument('--compute_metrics', default=False, type=bool,
                       help='If True metrics like mean and std will be computed')
    group.add_argument('--monte_carlo_samples', default=100, type=int,
                       help='Choose a number of monte carlo samples to compute statistical metrics')
    group.add_argument('--visualize', default=False, type=bool,
                       help="If True an image of a single generated result will be stored")

def train_args():
    """Define the Parser for the training"""
    
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_denoiser_options(parser)
    add_trainer_options(parser)
    add_training_options(parser)
    return parser.parse_args()


def inference_args():
    """Define the Parser for the inference"""

    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_denoiser_options(parser)
    add_trainer_options(parser)
    add_sde_options(parser)
    add_sampler_options(parser)
    add_metrics_options(parser)
    return parser.parse_args()