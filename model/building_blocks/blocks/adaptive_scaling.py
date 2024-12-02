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

from typing import Callable, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import default_init

Tensor = torch.Tensor

class AdaptiveScale(nn.Module):
  """Adaptively scale the input based on embedding.

  Conditional information is projected to two vectors of length c where c is
  the number of channels of x, then x is scaled channel-wise by first vector
  and offset channel-wise by the second vector.

  This method is now standard practice for conditioning with diffusion models,
  see e.g. https://arxiv.org/abs/2105.05233, and for the
  more general FiLM technique see https://arxiv.org/abs/1709.07871.
  """

  def __init__(self, 
               emb_channels: int,
               input_channels: int,
               input_dim: int,
               act_fun: Callable[[Tensor], Tensor]=F.silu, 
               dtype: torch.dtype=torch.float32,
               device: Any | None = None):
    super(AdaptiveScale, self).__init__()

    self.emb_channels = emb_channels
    self.input_channels = input_channels
    self.input_dim = input_dim
    self.act_fun = act_fun
    self.dtype = dtype
    self.device = device

    # self.affine = None
    self.affine = nn.Linear(
        in_features=emb_channels,
        out_features=input_channels * 2,
        dtype=self.dtype,
        device=self.device
      )
    default_init(1.0)(self.affine.weight)
    torch.nn.init.zeros_(self.affine.bias)

  def forward(self, x: Tensor, emb: Tensor) -> Tensor:
    """Adaptive scaling applied to the channel dimension.
    
    Args:
      x: Tensor to be rescaled.
      emb: Embedding values that drives the rescaling.

    Returns:
      Rescaled tensor plus bias
    """
    assert len(emb.shape) == 2, (
      "The dimension of the embedding needs to be two, instead it was : "
      + str(len(emb.shape))
    )

    scale_params = self.affine(self.act_fun(emb)) # (bs, c*2)

    scale, bias = torch.chunk(scale_params, 2, dim=-1)
    scale = scale[(...,) + (None,) * self.input_dim]
    bias = bias[(...,) + (None,) * self.input_dim]

    return x * (scale + 1) + bias