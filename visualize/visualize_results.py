import numpy as np
import pyvista as pv

from utils.visualization_utils import plotter_3d, gen_gt_plotter_3d

Array = np.ndarray

# Enable off-screen rendering
pv.OFF_SCREEN = True

# axis for analysis:
axis = 2 # y axis

# Load your data
data = np.load('solutions.npz')
gen_sample = data['gen_sample'] # gt_sample or gen_sample
gt_sample = data['gt_sample']


gen_gt_plotter_3d(gt_sample=gt_sample, gen_sample=gen_sample, axis=axis)


# MEAN AND STD OF ERROR
err_sample = np.abs(gt_sample[..., axis] - gen_sample[..., axis])
print(f"The mean error is: {np.mean(err_sample)}")
print(f"The standard deviation is: {np.std(err_sample)}")
