import os

import h5py
import netCDF4
import numpy as np
import skimage
from tqdm import tqdm

from decorators import timeit
from utilis_stats import StatsRecorderNew as StatsRecorder
from utils.dataloader_utils import downsample, translate_horizontally_periodic_unbatched, upsample
from dataloader.dataloader import DummyDataset


########################################################################################################################################
#### Base Class ####
########################################################################################################################################
class TrainingSetBase:
    def __init__(self,
                 size,
                 n_spatial_dim,
                 which,
                 training_samples,
                 learn_residual,
                 training_variable=None,
                 mean_training_=None,
                 std_training_=None,
                 start=0,
                 aug_data=0,
                 mc_samples=None,
                 workdir=None,
                 plot=False,
                 p_dropout=0,
                 scalar_norm=True):
        self.size = size
        self.n_spatial_dim = n_spatial_dim
        self.which = which
        self.start = start
        self.training_samples = training_samples
        self.aug_data = aug_data
        self.tot_samples = training_samples + aug_data if mc_samples is None else mc_samples
        self.learn_residual = learn_residual
        self.low_sizes = []
        self.training_variable = training_variable
        self.rand_gen = np.random.RandomState(seed=42)
        self.mean_training_ = mean_training_
        self.std_training_ = std_training_
        self.mean_training_input = None
        self.std_training_input = None
        self.mean_training_output = None
        self.std_training_output = None
        self.training_stats_path = workdir
        self.plot = plot
        self.input_channel = 1  # Placeholder redefine it in the child class
        self.output_channel = 1  # Placeholder redefine it in the child class
        self.stat_size = None
        self.p_dropout = 0
        self.use_scalar_norm = scalar_norm
        # self.use_low_res_stats = True  # Placeholder redefine it in the child class
        # self.use_scalar_norm = True
        print(f"Training Samples: {training_samples}. Starting from {start} with additional {aug_data} translational data")
        print(f"Total Samples: {self.tot_samples}")
        print(f"MC Samples: {mc_samples}")

    def __len__(self):
        return self.tot_samples

    def normalize_input(self, u_, ):
        if self.mean_training_input is not None:
            # return (u_ - self.mean_training_input) / (self.std_training_input + 1e-16)
            return (u_ - self.mean_training_input) / (self.std_training_input + 1e-16)
        else:
            return u_

    def denormalize_input(self, u_, ):

        if self.mean_training_input is not None:
            # return u_ * (self.std_training_input + 1e-16) + self.mean_training_input
            return u_ * (self.std_training_input + 1e-16) + self.mean_training_input
        else:
            return u_

    def normalize_output(self, u_, ):
        if self.mean_training_output is not None:
            # return (u_ - self.mean_training_output) / (self.std_training_output + 1e-16)
            return (u_ - self.mean_training_output) / (self.std_training_output + 1e-16)
        else:
            return u_

    def denormalize_output(self, u_, ):
        if self.mean_training_output is not None:
            # return u_ * (self.std_training_output + 1e-16) + self.mean_training_output
            return u_ * (self.std_training_output + 1e-16) + self.mean_training_output
        else:
            return u_

    def __getitem__(self, item):
        raise NotImplementedError()

    def get_index_shift(self, index):
        if index < self.training_samples or index > self.training_samples + self.aug_data:
            return index, 0
        else:
            return index % self.training_samples, self.rand_gen.randint(0, self.size)

    def get_proc_data(self, data):
        return data

    '''def collate(self):
        return utils_data.numpy_collate'''

    def collate_tf(self, data):
        return data

    @timeit
    def training_mean_var(self, workdir):

        if not os.path.isdir(self.training_stats_path):
            os.mkdir(self.training_stats_path)
        mean_path_1 = f"{self.training_stats_path}/mean_{self.training_samples}_{self.aug_data}_{self.size}_{self.learn_residual}.npy"
        mean_path_2 = f"{workdir}/mean_{self.training_samples}_{self.aug_data}_{self.size}_{self.learn_residual}.npy"

        std_path_1 = f"{self.training_stats_path}/std_{self.training_samples}_{self.aug_data}_{self.size}_{self.learn_residual}.npy"
        std_path_2 = f"{workdir}/std_{self.training_samples}_{self.aug_data}_{self.size}_{self.learn_residual}.npy"

        if os.path.isfile(mean_path_1):
            print("Found file for statistics: ", mean_path_1)
            means = np.load(mean_path_1, allow_pickle=True)
            stds = np.load(std_path_1, allow_pickle=True)

        elif os.path.isfile(mean_path_2):
            print("Found file for statistics: ", mean_path_2)
            means = np.load(mean_path_2, allow_pickle=True)
            stds = np.load(std_path_2, allow_pickle=True)
        else:
            print(f"Found no file at {mean_path_1} or {mean_path_2}. Proceed at computing it.")

            stats_recorder_input = StatsRecorder()
            stats_recorder_output = StatsRecorder()
            bs = 10
            print(self.__class__.__name__)
            Loader = JaxDataLoader if "Time" not in self.__class__.__name__ else JaxTimeLoader
            loader = Loader(self, n_spatial_dim=self.n_spatial_dim, batch_size=bs, shuffle=False, epochs=1).get_loader()

            # for kk in range(self.training_samples + self.aug_data):
            for data in tqdm(loader, total=len(self) // bs):
                # data = self[kk]

                u0 = self.get_proc_data(data)[..., :self.input_channel]  # [None, ...]
                u = self.get_proc_data(data)[..., self.input_channel:]  # [None, ...]
                stats_recorder_input.update(u0)
                stats_recorder_output.update(u)

            means = np.concatenate((stats_recorder_input.mean, stats_recorder_output.mean), -1)
            stds = np.concatenate((stats_recorder_input.std, stats_recorder_output.std), -1)
            np.save(mean_path_1, means)
            np.save(std_path_1, stds)
        return means, stds

    def transform(self, img):
        if self.which == "resize_wrap":
            resized_img = skimage.transform.resize(img, (self.size, self.size), order=3, anti_aliasing=True, mode="wrap")
        elif self.which == "nothing":
            resized_img = upsample(img.reshape(1, 1, img.shape[0], img.shape[1]), self.size)[0, 0]
        else:
            raise ValueError(f"Upsampling and downsampling not implemented for arg {self.which}")
        return resized_img

    def transform_batch(self, img):
        batch_size, nx, ny, channels = img.shape
        if self.which == "resize_wrap":
            resized_img = skimage.transform.resize(img, (batch_size, self.size, self.size, channels), order=3, anti_aliasing=True, mode="wrap")
        elif self.which == "nothing":
            resized_img = np.transpose(upsample(np.transpose(img, (0, 3, 1, 2)), self.size), (0, 2, 3, 1))
        else:
            raise ValueError(f"Upsampling and downsampling not implemented for arg {self.which}")
        return resized_img

    def downsample_batch(self, img, size):
        batch_size, nx, ny, channels = img.shape
        if self.which == "resize_wrap":
            resized_img = skimage.transform.resize(img, (batch_size, size, size, channels), anti_aliasing=False, order=3, mode="wrap")
        elif self.which == "nothing":
            resized_img = np.transpose(downsample(np.transpose(img, (0, 3, 1, 2)), size), (0, 2, 3, 1))
        else:
            raise ValueError(f"Upsampling and downsampling not implemented for arg {self.which}")
        return resized_img

    def downsample(self, img, size):
        if self.which == "resize_wrap":
            resized_img = skimage.transform.resize(img, (size, size), anti_aliasing=False, order=3, mode="wrap")
        elif self.which == "nothing":
            resized_img = downsample(img.reshape(1, 1, img.shape[0], img.shape[1]), size)[0, 0]
        else:
            raise ValueError(f"Upsampling and downsampling not implemented for arg {self.which}")
        return resized_img

    def get_true_stats_for_var_2(self, name, size, sample=99000):
        if not os.path.isfile(f"/cluster/work/math/camlab-data/data/diffusion_project/GroundTruthStats_{name}_{self.which}_{size}_{sample}/Stats.nc"):
            raise FileNotFoundError("Statistics file not found. Run ComputeTrueStatistics.py")
        f = netCDF4.Dataset(f"/cluster/work/math/camlab-data/data/diffusion_project/GroundTruthStats_{name}_{self.which}_{size}_{sample}/Stats.nc", 'r')
        mean = np.array(f.variables['mean_'][:])
        std = np.array(f.variables['std_'][:])
        spectrum = np.array(f.variables['spectrum_'][:])
        energy = np.array(f.variables['energy_'][:])
        idx_wass = np.array(f.variables['idx'][:])
        sol_wass = np.array(f.variables['sol_'][:])
        kk = np.arange(1, spectrum.shape[0] + 1)

        return mean, std, kk, spectrum, energy, idx_wass, sol_wass

    def get_true_stats_for_var_cloud(self, name, size, sample=99000):
        if not os.path.isfile(f"GroundTruthStats_{name}_{self.which}_{size}_{sample}/Stats.nc"):
            raise FileNotFoundError("Statistics file not found. Run ComputeTrueStatistics.py")
        f = netCDF4.Dataset(f"GroundTruthStats_{name}_{self.which}_{size}_{sample}/Stats.nc", 'r')
        mean = np.array(f.variables['mean_'][:])
        std = np.array(f.variables['std_'][:])
        spectrum = np.array(f.variables['spectrum_'][:])
        energy = np.array(f.variables['energy_'][:])
        idx_wass = np.array(f.variables['idx'][:])
        sol_wass = np.array(f.variables['sol_'][:])
        kk = np.arange(1, spectrum.shape[0] + 1)

        return mean, std, kk, spectrum, energy, idx_wass, sol_wass

    def set_mean_and_std(self, mean_training_, std_training_):
        self.mean_training_, self.std_training_ = mean_training_, std_training_
        self.mean_training_input, self.std_training_input = mean_training_[..., :self.input_channel], std_training_[..., :self.input_channel]
        self.mean_training_output, self.std_training_output = mean_training_[..., self.input_channel:], std_training_[..., self.input_channel:]

        if self.use_scalar_norm:
            self.mean_training_input = np.mean(self.mean_training_input, (0, 1))
            self.mean_training_output = np.mean(self.mean_training_output, (0, 1))

            self.std_training_input = np.mean(self.std_training_input ** 2, (0, 1)) ** 0.5
            self.std_training_output = np.mean(self.std_training_output ** 2, (0, 1)) ** 0.5

            print(self.mean_training_input, self.mean_training_output, self.std_training_input, self.std_training_output)
            
        '''else:

            self.std_training_input = np.mean(self.std_training_input ** 2, (0, 1)) ** 0.5
            self.std_training_output = np.mean(self.std_training_output ** 2, (0, 1)) ** 0.5'''

        print(f"Means shapes: {self.mean_training_input.shape}, {self.mean_training_output.shape}")
        print(f"Stds shapes {self.std_training_input.shape}, {self.std_training_output.shape}")

    def classifier_free_diffusion(self, inputs):
        p = self.rand_gen.random()
        if p < self.p_dropout:
            inputs[:] = 0
        return inputs



class DataIC_Vel(TrainingSetBase):
    def __init__(self, size, n_spatial_dim, which, training_samples, learn_residual, training_variable=None, mean_training_=None, std_training_=None, start=0, aug_data=0, mc_samples=None, workdir=None, plot=False, p_dropout=0, scalar_norm=True, compute_mean=True):
        super().__init__(size, n_spatial_dim, which, training_samples, learn_residual, training_variable, mean_training_, std_training_, start, aug_data, mc_samples, workdir, plot, p_dropout, scalar_norm)
        self.class_name = self.__class__.__name__
        self.input_channel = 2
        self.output_channel = 2
        self.stat_size = size
        self.use_low_res_stats = False
        self.p_dropout = p_dropout
        # self.chunks = {'member': 1000, 'time': -1, 'c': -1, 'x': -1, 'y': -1}
        self.ds = netCDF4.Dataset(f'/cluster/work/math/camlab-data/data/diffusion_project/ddsl_fast_{which}_{size}_tr2.nc', mode='r')

        if compute_mean:

            self.training_stats_path = f"/cluster/work/math/camlab-data/data/diffusion_project/TrainingStats_{self.which}_{self.class_name}"

            self.mean_down, self.std_down, self.kk, self.spectrum, self.energy, self.idx_wass, self.sol_wass = self.get_true_stats_for_var_2(self.class_name, self.size)
            self.std_data = np.sqrt(np.mean(self.std_down ** 2))
            if mean_training_ is None and std_training_ is None:
                self.mean_training_input, self.std_training_input = None, None
                self.mean_training_output, self.std_training_output = None, None
                mean_training_, std_training_ = self.training_mean_var(workdir=workdir)
            else:
                print("Mean and variance provided, using them. Shape:", mean_training_.shape, std_training_.shape)
            self.set_mean_and_std(mean_training_, std_training_)
        else:
            print("skipping")

    # @functools.lru_cache(maxsize=10000)
    def __getitem__(self, index):

        index += self.start

        (index, shift) = self.get_index_shift(index) if self.start == 0 else (index, 0)
        data = self.ds['data'][index].data

        data_input = data[0, ..., :self.input_channel]
        data_output = data[1, ..., :self.output_channel]

        data_input = self.normalize_input(data_input)

        if self.p_dropout != 0:
            data_input = self.classifier_free_diffusion(data_input)

        data_output = self.normalize_output(data_output)

        inputs = np.concatenate((data_input, data_output), -1)
        if shift != 0:
            inputs = translate_horizontally_periodic_unbatched(inputs, shift, axis=1)
        return inputs