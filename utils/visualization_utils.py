import torch
import numpy as np
import pyvista as pv
import os
import matplotlib.pyplot as plt
from typing import Union

array = np.ndarray
Tensor = torch.Tensor

# Enable off-screen rendering
pv.OFF_SCREEN = True


def load_data(
        file_name: str = 'solutions.npz', 
        save_dir: str = None
    ) -> tuple[array, array]:
    """Load Data from file stored after evaluation"""

    if save_dir is None:
        file = file_name
    else:
        file = os.path.join(save_dir, file_name)

    data = np.load(file)
    gen_sample = data['gen_sample']
    gt_sample = data['gt_sample']
    return (gen_sample, gt_sample)


def reshape_to_numpy(sample: Union[Tensor, array]) -> Union[Tensor, array]:
    """Converts a tensor or array to a NumPy array with appropriate dimension ordering."""

    if Tensor and isinstance(sample, Tensor):
        if sample.ndim == 3:
            return sample.permute(1, 2, 0).cpu().numpy()
        elif sample.ndim == 4:
            return sample.permute(1, 2, 3, 0).cpu().numpy()
        else:
            raise ValueError(f'sample dim should be 3 or 4 and not {sample.ndim}')
    elif isinstance(sample, array):
        if sample.ndim == 3:
            return sample.transpose(1, 2, 0)
        elif sample.ndim == 4:
            return sample.transpose(1, 2, 3, 0)
        else:
            raise ValueError(f'sample dim should be 3 or 4 and not {sample.ndim}')
    else:
        raise TypeError("Input must be a numpy array or PyTorch tensor.")
    

def check_save_dir(save_dir: str = None):
    """Checks whether the given path exists to save the image"""

    if save_dir is None:
        raise ValueError("To store results provide an existing path.")
    elif save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)


def plot_2d_sample(
        gen_sample: Union[Tensor, array], 
        gt_sample: Union[Tensor, array], 
        axis: int = 0,
        save: bool = True,
        save_dir: str = None
    ):
    """Plots the 2D results"""

    # check whether path exists
    if save == True:
        check_save_dir(save_dir)

    gen_sample = reshape_to_numpy(gen_sample)
    gt_sample = reshape_to_numpy(gt_sample)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(gen_sample[..., axis])
    axes[0].set_title("Generated")
    axes[0].axis('off')

    axes[1].imshow(gt_sample[..., axis])
    axes[1].set_title("Groundtruth")
    axes[1].axis('off')

    plt.tight_layout()

    if save:
        save_path = os.path.join(save_dir, "gen_gt_sample.png")
        plt.savefig(save_path)
    else:
        plt.show()


def plotter_3d(
        sample: Union[array, Tensor], 
        axis: int=0, 
        save: bool = True, 
        save_dir: str = None
    ):
    """3D plotter to visualize generated or ground truth 3D data"""

    if save == True:
        check_save_dir(save_dir)

    sample = reshape_to_numpy(sample)

    volume = pv.wrap(sample[..., axis])
    plotter = pv.Plotter(off_screen=True)
    plotter.add_volume(volume, opacity="sigmoid", cmap="viridis", shade=True)
    if save:
        save_path = os.path.join(save_dir, "gen_sample.png")
        plotter.screenshot(save_path)
    else:
        plotter.show()
    plotter.close()


def gen_gt_plotter_3d(
        gt_sample: array, 
        gen_sample: array, 
        axis: int=0, 
        save: bool = True,
        save_dir: str = None):
    """3D plotter to visualize generated and ground truth 3D data side by side"""

    if save == True:
        check_save_dir(save_dir)

    gt_sample = reshape_to_numpy(gt_sample)
    gen_sample = reshape_to_numpy(gen_sample)

    volume_gen = pv.wrap(gen_sample[..., axis])
    volume_gt = pv.wrap(gt_sample[..., axis])

    # Set up the plotter with two viewports side by side
    plotter = pv.Plotter(off_screen=True, shape=(1, 2))

    plotter.subplot(0, 0)
    plotter.add_volume(volume_gen, opacity="sigmoid", cmap="viridis", shade=True, show_scalar_bar=False)
    plotter.add_text("Generated Sample", position='upper_edge', font_size=12, color='black')

    plotter.subplot(0, 1)
    plotter.add_volume(volume_gt, opacity="sigmoid", cmap="viridis", shade=True, show_scalar_bar=False)
    plotter.add_text("Ground Truth Sample", position='upper_edge', font_size=12, color='black')

    if save:
        save_path = os.path.join(save_dir, "gen_gt_sample.png")
        plotter.screenshot(save_path)
    else:
        plotter.show()
    plotter.close()
