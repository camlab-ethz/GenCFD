# Copyright 2024 The swirl_dynamics Authors.
# Modifications made by the CAM Lab at ETH Zurich.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modules for diffusion models."""

from __future__ import annotations
import dataclasses
from typing import Union, Callable, Protocol
import torch as th
import numpy as np

Tensor = th.Tensor
Numeric = Union[bool, int, float, np.ndarray, th.Tensor]
ScheduleFn = Callable[[Numeric], Numeric]

EPS = 1e-4
MIN_DIFFUSION_TIME = EPS
MAX_DIFFUSION_TIME = 1.0 - EPS


@dataclasses.dataclass(frozen=True)
class InvertibleSchedule:
    """An invertible schedule.

    The schedule consists of a forward function that maps diffusion times to
    noise/scale/logsnr values, and an inverse that maps noise/scale/logsnr
    values back to the corresponding diffusion times, such that
    t = inverse(forward(t)). These functions should be monotonic wrt their input
    argument.

    Attributes:
      forward: A monotonic schedule function that maps diffusion times (scalar or
        array) to the noise/scale/logsnr values.
      inverse: The inverse schedule that maps scheduled values (scalar or array)
        back to the corresponding diffusion times.
    """

    forward: ScheduleFn
    inverse: ScheduleFn

    def __call__(self, t: Numeric) -> Numeric:
        return self.forward(t)


def sigma2logsnr(
    sigma: InvertibleSchedule, device: th.device = None
) -> InvertibleSchedule:
    """Converts a sigma schedule to a logsnr schedule."""
    forward = lambda t: -2 * th.log(th.as_tensor(sigma(t), device=device))
    inverse = lambda logsnr: sigma.inverse(
        th.exp(th.as_tensor(-logsnr / 2), device=device)
    )
    return InvertibleSchedule(forward, inverse)


def logsnr2sigma(
    logsnr: InvertibleSchedule, device: th.device = None
) -> InvertibleSchedule:
    """Converts a logsnr schedule to a sigma schedule."""
    forward = lambda t: th.exp(th.as_tensor(-logsnr(t) / 2, device=device))
    inverse = lambda sigma: logsnr.inverse(
        -2 * th.log(th.as_tensor(sigma, device=device))
    )
    return InvertibleSchedule(forward, inverse)


@dataclasses.dataclass(frozen=True)
class Diffusion:
    """Diffusion scheme.

    Fully parametrizes the Gaussian perturbation kernel:

      p(x_t|x_0) = N(x_t; s_t * x_0, s_t * σ_t * I)

    where x_0 and x_t are original and noised samples. s_t and σ_t are the scale
    and noise schedules. t ∈ [0, 1]. I denotes the identity matrix. This
    particular parametrization follows Karras et al.
    (https://arxiv.org/abs/2206.00364).

    Attributes:
      scale: The scale schedule (as a function of t).
      sigma: The noise schedule (as monotonically increasing function of t).
      logsnr: The log signal-to-noise (LogSNR) schedule equivalent to the sigma
        schedule.
      sigma_max: The maximum noise level of the scheme.
      device: The device on which the tensor will be placed (e.g. 'cuda', 'cpu')
    """

    scale: ScheduleFn
    sigma: InvertibleSchedule

    @property
    def logsnr(self, device: th.device = None) -> InvertibleSchedule:
        return logsnr2sigma(self.sigma, device)

    @property
    def sigma_max(self) -> Numeric:
        return self.sigma(MAX_DIFFUSION_TIME)

    @classmethod
    def create_variance_preserving(
        cls, sigma: InvertibleSchedule, data_std: float = 1.0, device: th.device = None
    ) -> Diffusion:
        """Creates a variance preserving diffusion scheme.

        Derive the scale schedule s_t from the noise schedule σ_t such that
        s_t^2 * (σ_d^2 + σ_t^2) (where σ_d denotes data standard deviation) remains
        constant (at σ_d^2) for all t. See Song et al.
        (https://arxiv.org/abs/2011.13456) for reference.

        Args:
          sigma: The sigma (noise) schedule.
          data_std: The standard deviation (scalar) of the data.

        Returns:
          A variance preserving diffusion scheme.
        """
        var = th.square(th.as_tensor(data_std, device=device))
        scale = lambda t: th.sqrt(
            var / (var + th.square(th.as_tensor(sigma(t), device=device)))
        )
        return cls(scale=scale, sigma=sigma)

    @classmethod
    def create_variance_exploding(
        cls,
        sigma: InvertibleSchedule,
        data_std: float = 1.0,
        device: th.device | None = None,
    ) -> Diffusion:
        """Creates a variance exploding diffusion scheme.

        Scale s_t is kept constant at 1. The noise schedule is scaled by the
        data standard deviation such that the amount of noise added is proportional
        to the data variation. See Song et al.
        (https://arxiv.org/abs/2011.13456) for reference.

        Args:
          sigma: The sigma (noise) schedule.
          data_std: The standard deviation (scalar) of the data.

        Returns:
          A variance exploding diffusion scheme.
        """
        scaled_forward = lambda t: th.as_tensor(sigma(t), device=device) * data_std
        scaled_inverse = lambda y: th.as_tensor(
            sigma.inverse(y / data_std), device=device
        )
        scaled_sigma = InvertibleSchedule(scaled_forward, scaled_inverse)
        return cls(
            scale=lambda s: th.as_tensor(s / s, device=device), sigma=scaled_sigma
        )


def create_variance_preserving_scheme(
    sigma: InvertibleSchedule, data_std: float = 1.0, device: th.device = None
) -> Diffusion:
    """Alias for `Diffusion.create_variance_preserving`."""
    return Diffusion.create_variance_preserving(sigma, data_std, device)


def create_variance_exploding_scheme(
    sigma: InvertibleSchedule, data_std: float = 1.0, device: th.device = None
) -> Diffusion:
    """Alias for `Diffusion.create_variance_exploding`."""
    return Diffusion.create_variance_exploding(sigma, data_std, device)


def _linear_rescale(
    in_min: float,
    in_max: float,
    out_min: float,
    out_max: float,
) -> InvertibleSchedule:
    """Linearly rescale input between specified ranges."""
    in_range = in_max - in_min
    out_range = out_max - out_min
    fwd = lambda x: out_min + (x - in_min) / in_range * out_range
    inv = lambda y: in_min + (y - out_min) / out_range * in_range
    return InvertibleSchedule(fwd, inv)


def tangent_noise_schedule(
    clip_max: float = 100.0,
    start: float = 0.0,
    end: float = 1.5,
    device: th.device = None,
) -> InvertibleSchedule:
    """Tangent noise schedule.

    This schedule is obtained by taking the section of the tan(t) function
    inside domain [`start`, `end`], and applying linear rescaling such that the
    input domain is [0, 1] and output range is [0, `clip_max`].

    This is really the "cosine" schedule proposed in Dhariwal and Nicol
    (https://arxiv.org/abs/2105.05233). The original schedule is
    a cosine function in γ, i.e. γ = cos(pi/2 * t) for t in [0, 1]. With
    γ = 1 / (σ^2 + 1), the corresponding σ schedule is a tangent function.

    The "shifted" cosine schedule proposed in Hoogeboom et al.
    (https://arxiv.org/abs/2301.11093) simply corresponds to adjusting
    the `clip_max` parameter. Empirical evidence suggests that one should consider
    increasing this maximum noise level when modeling higher resolution images.

    Args:
      clip_max: The maximum noise level in the schedule.
      start: The left endpoint of the tangent function domain used.
      end: The right endpoint of the tangent function domain used.

    Returns:
      A tangent noise schedule.
    """
    if not -th.pi / 2 < start < end < th.pi / 2:
        raise ValueError("Must have -pi/2 < `start` < `end` < pi/2.")

    in_rescale = _linear_rescale(
        in_min=MIN_DIFFUSION_TIME, in_max=MAX_DIFFUSION_TIME, out_min=start, out_max=end
    )
    out_rescale = _linear_rescale(
        in_min=np.tan(start), in_max=np.tan(end), out_min=0.0, out_max=clip_max
    )

    sigma = lambda t: out_rescale(th.tan(th.as_tensor(in_rescale(t), device=device)))
    inverse = lambda y: in_rescale.inverse(
        th.arctan(th.as_tensor(out_rescale.inverse(y), device=device))
    )

    return InvertibleSchedule(sigma, inverse)


def power_noise_schedule(
    clip_max: float = 100.0,
    p: float = 1.0,
    start: float = 0.0,
    end: float = 1.0,
    device: th.device = None,
) -> InvertibleSchedule:
    """Power noise schedule.

    This schedule is obtained by taking the section of the t^p (where p > 0)
    function inside domain [`start`, `end`], and applying linear rescaling such
    that the input domain is [0, 1] and output range is [0, `clip_max`].

    Variance exploding schedules in Karras et al.
    (https://arxiv.org/abs/2206.00364) and Song et al.
    (https://arxiv.org/abs/2011.13456) use p = 1 and p = 0.5 respectively.

    Args:
      clip_max: The maximum noise level in the schedule.
      p: The degree of power schedule.
      start: The left endpoint of the power function domain used.
      end: The right endpoint of the power function domain used.

    Returns:
      A power noise schedule.
    """
    if not (0 <= start < end and p > 0):
        raise ValueError("Must have `p` > 0 and 0 <= `start` < `end`.")

    in_rescale = _linear_rescale(
        in_min=MIN_DIFFUSION_TIME, in_max=MAX_DIFFUSION_TIME, out_min=start, out_max=end
    )
    out_rescale = _linear_rescale(
        in_min=start**p, in_max=end**p, out_min=0.0, out_max=clip_max
    )

    sigma = lambda t: out_rescale(th.pow(th.as_tensor(in_rescale(t), device=device), p))
    inverse = lambda y: in_rescale.inverse(  # pylint:disable=g-long-lambda
        th.pow(th.as_tensor(out_rescale.inverse(y), device=device), 1 / p)
    )

    return InvertibleSchedule(sigma, inverse)


def exponential_noise_schedule(
    clip_max: float = 100.0,
    base: float = np.e**0.5,
    start: float = 0.0,
    end: float = 5.0,
    device: th.device = None,
) -> InvertibleSchedule:
    """Exponential noise schedule.

    This schedule is obtained by taking the section of the base^t (where base > 1)
    function inside domain [`start`, `end`], and applying linear rescaling such
    that the input domain is [0, 1] and output range is [0, `clip_max`]. This
    schedule is always a convex function.

    Args:
      clip_max: The maximum noise level in the schedule.
      base: The base of the exponential. Defaults to sqrt(e) so that σ^2 follows
        schedule exp(t).
      start: The left endpoint of the exponential function domain used.
      end: The right endpoint of the exponential function domain used.

    Returns:
      An exponential noise schedule.
    """
    if not (start < end and base > 1.0):
        raise ValueError("Must have `base` > 1 and `start` < `end`.")

    in_rescale = _linear_rescale(
        in_min=MIN_DIFFUSION_TIME,
        in_max=MAX_DIFFUSION_TIME,
        out_min=start,
        out_max=end,
    )
    out_rescale = _linear_rescale(
        in_min=base**start, in_max=base**end, out_min=0.0, out_max=clip_max
    )

    sigma = lambda t: out_rescale(
        th.pow(th.as_tensor(base, device=device), in_rescale(t))
    )
    inverse = lambda y: in_rescale.inverse(  # pylint:disable=g-long-lambda
        th.log(th.as_tensor(out_rescale.inverse(y), device=device))
        / th.log(th.as_tensor(base, device=device))
    )
    return InvertibleSchedule(sigma, inverse)


# ********************
# Noise sampling
# ********************


class NoiseLevelSampling(Protocol):

    def __call__(self, shape: tuple[int, ...]) -> Tensor:
        """Samples noise levels for training."""
        ...


def _uniform_samples(
    shape: tuple[int, ...], uniform_grid: bool, device: th.device = None
) -> th.tensor:
    """Generates samples from uniform distribution on [0, 1]."""
    if uniform_grid:
        s0 = th.rand((), dtype=th.float32, device=device)
        num_elements = int(np.prod(shape))
        step_size = 1 / num_elements
        grid = th.linspace(
            0, 1 - step_size, num_elements, dtype=th.float32, device=device
        )
        samples = th.remainder(grid + s0, 1).reshape(shape)
    else:
        samples = th.rand(shape, dtpye=th.float32, device=device)
    return samples


def log_uniform_sampling(
    scheme: Diffusion,
    clip_min: float = 1e-4,
    uniform_grid: bool = False,
    device: th.device = None,
) -> NoiseLevelSampling:
    """Samples noise whose natural log follows a uniform distribution."""

    def _noise_sampling(shape: tuple[int, ...]) -> Tensor:
        samples = _uniform_samples(shape, uniform_grid, device)
        log_min = th.log(th.as_tensor(clip_min, dtype=samples.dtype, device=device))
        log_max = th.log(
            th.as_tensor(scheme.sigma_max, dtype=samples.dtype, device=device)
        )
        samples = (log_max - log_min) * samples + log_min
        return th.exp(samples)

    return _noise_sampling


def time_uniform_sampling(
    scheme: Diffusion,
    clip_min: float = 1e-4,
    uniform_grid: bool = False,
    device: th.device = None,
) -> NoiseLevelSampling:
    """Samples noise from a uniform distribution in t."""

    def _noise_sampling(shape: tuple[int, ...]) -> Tensor:
        samples = _uniform_samples(shape, uniform_grid, device=device)
        min_t = scheme.sigma.inverse(clip_min)
        samples = (MAX_DIFFUSION_TIME - min_t) * samples + min_t
        return th.as_tensor(scheme.sigma(samples), device=device)

    return _noise_sampling


def normal_sampling(
    scheme: Diffusion,
    clip_min: float = 1e-4,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    device: th.device = None,
) -> NoiseLevelSampling:
    """Samples noise from a normal distribution.

    This noise sampling is first used in Karras et al.
    (https://arxiv.org/abs/2206.00364). The default mean and standard deviation
    settings are designed for diffusion scheme with sigma_max = 80.

    Args:
      scheme: The diffusion scheme.
      clip_min: The minimum noise cutoff.
      p_mean: The mean of the sampling normal distribution.
      p_std: The standard deviation of the sampling normal distribution.

    Returns:
      A normal sampling function.
    """

    def _noise_sampler(shape: tuple[int, ...]) -> Tensor:
        log_sigma = th.normal(
            mean=0, std=1, size=shape, dtype=th.float32, device=device
        )
        log_sigma = p_mean + p_std * log_sigma
        return th.clamp(th.exp(log_sigma), min=clip_min, max=scheme.sigma_max)

    return _noise_sampler


# ********************
# Noise weighting
# ********************


class NoiseLossWeighting(Protocol):

    def __call__(self, sigma: Tensor) -> Tensor:
        """Returns weights of the input noise levels in the loss function."""
        ...


def inverse_squared_weighting(sigma: Tensor) -> Tensor:
    return 1 / th.square(sigma)


def edm_weighting(
    data_std: float = 1.0, device: th.device = None
) -> NoiseLossWeighting:
    """Weighting proposed in Karras et al. (https://arxiv.org/abs/2206.00364).

    This weighting ensures the effective weights are uniform across noise levels
    (see appendix B.6, eqns 139 to 144).

    Args:
      data_std: the standard deviation of the data.

    Returns:
      The weighting function.
    """

    def _weight_fn(sigma: Tensor) -> Tensor:
        return (
            th.square(th.tensor(data_std, device=device)) + th.square(sigma)
        ) / th.square(data_std * sigma)

    return _weight_fn


def likelihood_weighting(
    self, data_std: float = 1.0, scheme: Diffusion = None
) -> NoiseLossWeighting:
    """
    Weighting proposed by Song et al. (https://arxiv.org/abs/2101.09258).

    Args:
        data_std: the standard deviation of the data.
        scheme: the diffusion scheme

    Returns:
        the weighting function.
    """

    def _weight_fn(sigma: Tensor) -> Tensor:
        t = scheme.sigma.inverse(sigma)
        s = scheme.scale(t)  # scaling factor s

        def vmap_dsquare_dt(f: ScheduleFn) -> ScheduleFn:
            """
            Returns d/dt (f(t))^2 = 2ḟ(t)f(t) given f(t).
            """

            def grad_fn(t: Tensor) -> Tensor:
                t.requires_grad_(True)
                f_t = f(t)
                f_square = th.square(f_t)  # f(t)^2
                grad_f_square = th.autograd.grad(f_square.sum(), t, create_graph=True)[
                    0
                ]
                return grad_f_square

            # Assuming `t` is a batched tensor
            return grad_fn

        dsquare_sigma_dt = vmap_dsquare_dt(scheme.sigma)(
            t
        )  # dsigma^2/dt using vmap_dsquare_dt
        lambda_ = dsquare_sigma_dt * th.square(s)  # Lambda (loss weighting)

        return lambda_

    return _weight_fn
