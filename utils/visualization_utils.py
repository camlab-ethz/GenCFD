import numpy as np
import pyvista as pv
import os

array = np.ndarray

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


def plotter_3d(sample: array, axis: int=0):
    """3D plotter to visualize generated or ground truth 3D data"""

    volume = pv.wrap(sample[..., axis])
    plotter = pv.Plotter(off_screen=True)
    plotter.add_volume(volume, opacity="sigmoid", cmap="viridis", shade=True)
    plotter.screenshot("gen_3d_image.png")
    plotter.close()


def gen_gt_plotter_3d(
        gt_sample: array, 
        gen_sample: array, 
        axis: int=0, 
        save: bool = True):
    """3D plotter to visualize generated and ground truth 3D data side by side"""

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
        plotter.screenshot("result.png")
    plotter.close()
