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

"""Modules for guidance transforms for denoising functions."""

from typing import Any, Callable, Dict, Sequence, Literal, Protocol

import torch as th
from torch.autograd import grad

Tensor = th.Tensor
PyTree = Any
TensorMapping = Dict[str, Tensor]
DenoiseFn = Callable[[Tensor, Tensor, TensorMapping | None], Tensor]


class Transform(Protocol):
    """Transforms a denoising function to follow some guidance.

    One may think of these transforms as instances of Python decorators,
    specifically made for denoising functions. Each transform takes a base
    denoising function and extends it (often using some additional data) to build
    a new denoising function with the same interface.
    """

    def __call__(
        self, denoise_fn: DenoiseFn, guidance_inputs: TensorMapping
    ) -> DenoiseFn:
        """Constructs a guided denoising function.

        Args:
          denoise_fn: The base denoising function.
          guidance_inputs: A dictionary containing inputs used to construct the
            guided denoising function. Note that all transforms *share the same
            input dict*, therefore all transforms should use different fields from
            this dict (unless absolutely intended) to avoid potential name clashes.

        Returns:
          The guided denoising function.
        """
        ...


class InfillFromSlices:
    """N-dimensional infilling guided by known values on slices.

    Example usage::

      # 2D infill given every 8th pixel along both dimensions (assuming that the
      # lead dimension is for batch).
      slices = tuple(slice(None), slice(None, None, 8), slice(None, None, 8))
      sr_guidance = InfillFromSlices(slices, guide_strength=0.1)

      # Post-process a trained denoiser function via function composition.
      # The `observed_slices` arg must have compatible shape such that
      # `image[slices] = observed_slices` would not raise errors.
      guided_denoiser = sr_guidance(denoiser, {"observed_slices": jnp.Tensor(0.0)})

      # Run guided denoiser the same way as a normal one
      denoised = guided_denoiser(noised, sigma=jnp.Tensor(0.1), cond=None)

    Attributes:
      slices: The slices of the input to guide denoising (i.e. the rest is being
        infilled).
      guide_strength: The strength of the guidance relative to the raw denoiser.
        It will be rescaled based on the fraction of values being conditioned.
    """

    def __init__(self, slices: tuple[slice, ...], guide_strength: float = 0.5):
        self.slices = slices
        self.guide_strength = guide_strength

    def __call__(
        self, denoise_fn: DenoiseFn, guidance_inputs: TensorMapping
    ) -> DenoiseFn:
        """Constructs denoise function guided by values on specified slices."""

        def _guided_denoise(
            x: Tensor, sigma: Tensor, cond: TensorMapping | None = None
        ) -> Tensor:
            def constraint(xt: Tensor) -> tuple[Tensor, Tensor]:
                denoised = denoise_fn(xt, sigma, cond)
                error = th.sum(
                    (denoised[self.slices] - guidance_inputs["observed_slices"]) ** 2
                )

                return error, denoised

            x = x.requires_grad_(True)
            error, denoised = constraint(x)

            constraint_grad = grad(error, x, retain_graph=True)[0]
            # Rescale based on the fraction of values being conditioned.
            cond_fraction = th.prod(th.tensor(x[self.slices].shape)) / th.prod(
                th.tensor(x.shape)
            )
            guide_strength = self.guide_strength / cond_fraction
            denoised = denoised - guide_strength * constraint_grad
            denoised[self.slices] = guidance_inputs["observed_slices"]
            return denoised

        return _guided_denoise


class InterlockingFrames:
    """Condition on the first and last frame to be equal in a short trajectory.

    The main point of this guidance term is to interlock contigous temporal chunks
    into a larger one by imposing boundary conditions at the interfaces of the
    chunks. Each chunk is a short temporal sequence produced by a diffusion model
    with a given batch size. To ensure the spatio-temporal coherence the boundary
    conditions are imposed at each step of the evolution of the SDE in diffusion
    time.

    For a target number of frames to generate, total_num_frames, we decompose the
    total sequence in several chunks (num_chunks), whose number is given by the
    temporal generation length of the backbone model (minus the overlap).
    Then the batch dimension of the backbone diffusion model is fixed to
    num_chunks. Thus we generate num_chunks of short trajectories simultaneously,
    and we concatenate them (removing the overlapping regions).

    We can generate more than one long sequence, thus that input of the guidence
    is a 5-tensor with dimensions:
    (batch_size, num_chunks, num_frames_per_chunk, height, width, channels).

    Attributes:
      guide_strength: Strength of the conditioning relative to unconditioned
        score.
      style: How the boundaries are imposed following either "mean" or "swap". For
        "mean" we compute the mean between the overlap of two adjacent chunks,
        whereas for swap we exchange the values within the overlapping region.
        The rationale for the second is to not change the statistics by averaging
        two Gaussians.
      overlap_length: The length of the overlap which we impose to be the same
        across the boundaries.
    """

    def __init__(
        self,
        guide_strength: float = 0.5,
        style: Literal["average", "swap"] = "average",
        overlap_length: int = 1,
    ):
        self.guide_strength = guide_strength
        self.style = style
        self.overlap_length = overlap_length

    def __call__(
        self, denoise_fn: DenoiseFn, guidance_inputs: TensorMapping | None = None
    ) -> DenoiseFn:
        """Constructs denoise function conditioned on overlaping values."""

        def _guided_denoise(
            x: Tensor, sigma: Tensor, cond: TensorMapping | None = None
        ) -> Tensor:
            """Guided denoise function.

            Args:
              x: The tensor to denoise with dims (num_trajectories, num_chunks_traj,
              num_frames_per_chunk, height, width, channels).
              sigma: The noise level.
              cond: The dictionary with the conditioning.

            Returns:
              An estimate of the denoised tensor guided by the interlocking
              constraint.
            """
            if x.ndim != 6:
                raise ValueError(
                    f"Invalid input dimension: {x.shape}, a 6-tensor is " "expected."
                )

            def constraint(xt: Tensor) -> tuple[Tensor, Tensor]:
                denoised = denoise_fn(xt, sigma, cond)
                error = th.sum(
                    (
                        denoised[:, 1:, : self.overlap_length]
                        - denoised[:, :-1, -self.overlap_length :]
                    )
                    ** 2
                )
                return error, denoised

            x = x.requires_grad_(True)
            error, denoised = constraint(x)

            constraint_grad = grad(error, x, retain_graph=True, allow_unused=True)[0]
            denoised = denoised - self.guide_strength * constraint_grad

            # Interchanging information at each side of the interface.
            if self.style not in ["average", "swap"]:
                raise ValueError(
                    f"Invalid style: {self.style}. Expected either"
                    '"average" or "swap".'
                )
            elif self.style == "swap":
                cond_value_first = denoised[:, 1:, : self.overlap_length].clone()
                denoised[:, 1:, : self.overlap_length] = denoised[
                    :, :-1, -self.overlap_length :
                ]
                denoised[:, :-1, -self.overlap_length :] = cond_value_first

            # Average the values at each side of the interface.
            elif self.style == "average":
                average_value = 0.5 * (
                    denoised[:, 1:, : self.overlap_length]
                    + denoised[:, :-1, -self.overlap_length :]
                )
                denoised[:, 1:, : self.overlap_length] = average_value
                denoised[:, :-1, -self.overlap_length :] = average_value

            return denoised

        return _guided_denoise


class ClassifierFreeHybrid:
    """Classifier-free guidance for a hybrid (cond/uncond) denoising model.

    This guidance technique, introduced by Ho and Salimans
    (https://arxiv.org/abs/2207.12598), aims to improve the quality of denoised
    images by combining conditional and unconditional denoising outputs.
    The guided denoise function is given by:

      D̃(x, σ, c) = (1 + w) * D(x, σ, c) - w * D(x, σ, Ø),

    where

      - x: The noisy input.
      - σ: The noise level.
      - c: The conditioning information (e.g., class label).
      - Ø: A special masking condition (typically zeros) indicating unconditional
        denoising.
      - w: The guidance strength, controlling the influence of each denoising
        output. A value of 0 indicates non-guided denoising.

    Attributes:
      guidance_strength: The strength of guidance (i.e. w). The original paper
        reports optimal values of 0.1 and 0.3 for 64x64 and 128x128 ImageNet
        respectively.
      cond_mask_keys: A collection of keys in the conditions dictionary that will
        be masked. If `None`, all conditions are masked.
      cond_mask_value: The values that the conditions will be masked by. This
        value must be consistent with the masking applied at training.
    """

    def __init__(
        self,
        guidance_strength: float = 0.0,
        cond_mask_keys: Sequence[str] | None = None,
        cond_mask_value: float = 0.0,
    ):
        self.guidance_strength = guidance_strength
        self.cond_mask_value = cond_mask_value
        self.cond_mask_keys = cond_mask_keys

    def __call__(
        self, denoise_fn: DenoiseFn, guidance_inputs: TensorMapping
    ) -> DenoiseFn:
        """Constructs denoise function with classifier free guidance."""

        def _guided_denoise(
            x: Tensor, sigma: Tensor, cond: TensorMapping, y: Tensor = None
        ) -> Tensor:
            masked_cond = {
                k: (
                    v  # pylint: disable=g-long-ternary
                    if self.cond_mask_keys is not None and k not in self.cond_mask_keys
                    else th.ones_like(v) * self.cond_mask_value
                )
                for k, v in cond.items()
            }
            uncond_denoised = denoise_fn(x, sigma, masked_cond)
            denoised = (1 + self.guidance_strength) * denoise_fn(
                x, sigma, cond
            ) - self.guidance_strength * uncond_denoised

            return denoised

        return _guided_denoise
