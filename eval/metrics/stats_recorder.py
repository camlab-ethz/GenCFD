import torch

Tensor = torch.Tensor

class StatsRecorder:
    """StatsRecorder which keeps track of metrics for the Ground Truth 
    and the Generated Data
    """
    def __init__(self, 
                 batch_size: int, 
                 ndim: int, 
                 channels: int, 
                 data_shape: tuple,
                 device: torch.device = None
        ):
        self.device = device

        # information about the dataset
        self.batch_size = batch_size
        self.ndim = ndim # 3D or 2D dataset
        self.axis = self._set_axis(ndim) # axis for channel wise mean
        self.channels = channels
        self.data_shape = data_shape

        # initialize mean and std
        self.mean_gt = torch.zeros(data_shape, device=device)
        self.mean_gen = torch.zeros(data_shape, device=device)
        self.std_gt = torch.zeros(data_shape, device=device)
        self.std_gen = torch.zeros(data_shape, device=device)

        # track the number of observation
        self.observation = 0


    def _set_axis(self, ndim: int):
        """Sets the axis for mean and std deviation based on ndim."""

        if ndim == 2:
            return (1, 2)
        elif ndim == 3:
            return (1, 2, 3)
        else:
            raise ValueError(f"Only 2D or 3D data supported, got {ndim}D data.")
        
    
    def _validate_data(self, gen_data: Tensor, gt_data: Tensor):
        """Validates the dimensionality and types of gen_data and gt_data."""

        if not (isinstance(gen_data, Tensor) and isinstance(gt_data, Tensor)):
            raise TypeError("gen_data and gt_data must be PyTorch Tensors.")
        
        expected_dim = 4 if self.ndim == 2 else 5
        assert gen_data.ndim == expected_dim and gt_data.ndim == expected_dim, (
            f"Expected shape (bs, c, x, y) for 2D or (bs, c, x, y, z) for 3D,"
            f"got gen_data {gen_data.shape} and gt_data {gt_data.shape}."
        )


    def update_step(self, gen_data: Tensor, gt_data: Tensor):
        """Updates all relevant metrics"""

        self._validate_data(gen_data, gt_data)
        self.update_step_mean_and_std(gen_data, gt_data)
        self.observation += self.batch_size


    def update_step_mean_and_std(self, gen_data: Tensor, gt_data: Tensor):
        """Keeps track of the mean of the dataset
        
        Instead of computing the mean of the error, the ground truth
        and the generated results, this method updates the mean of the 
        dataset itself for the gt as well as for the generated data
        """

        m = self.observation # current number of observations
        n = self.batch_size # observations added
        # new number of observations would then be m + n
        mean_gt_data = gt_data.mean(dim=0)
        mean_gen_data = gen_data.mean(dim=0)
        std_gt_data = gt_data.std(dim=0) if gt_data.size(0) > 1 else gt_data.std(dim=0, correction=0)
        std_gen_data = gen_data.std(dim=0) if gen_data.size(0) > 1 else gen_data.std(dim=0, correction=0)

        # first we will update the standard deviation before we update the mean
        var_gt = (
            (m / (m + n)) * (self.std_gt ** 2) + 
            (n / (m + n)) * (std_gt_data ** 2) + 
            (m * n / (m + n) ** 2) * (self.mean_gt - mean_gt_data) ** 2
        )
        self.std_gt = torch.sqrt(var_gt)

        var_gen = (
            (m / (m + n)) * (self.std_gen ** 2) + 
            (n / (m + n)) * (std_gen_data ** 2) + 
            (m * n / (m + n) ** 2) * (self.mean_gen - mean_gen_data) ** 2
        )
        self.std_gen = torch.sqrt(var_gen)

        # update the mean values
        self.mean_gt = m / (m + n) * self.mean_gt + n / (m + n) * mean_gt_data
        self.mean_gen = m / (m + n) * self.mean_gen + n / (m + n) * mean_gen_data
        
        