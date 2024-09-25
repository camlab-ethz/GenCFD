from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import reshape_jax_torch

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

  def __init__(self, act_fun: Callable[[Tensor], Tensor]=F.silu, 
               dtype: torch.dtype=torch.float32):
    super(AdaptiveScale, self).__init__()

    self.act_fun = act_fun
    self.dtype = dtype

    self.affine = None

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
    if self.affine is None:
      # Initialize affine transformation
      self.affine = nn.Linear(
        in_features=emb.shape[-1],
        out_features=reshape_jax_torch(x).shape[-1] * 2,
        dtype=self.dtype
      )

    scale_params = self.affine(self.act_fun(emb)) # (bs, c*2)
    # Unsqueeze in the middle to allow broadcasting. 
    # 2D case: (bs, 1, 1, c*2), 3D case: (bs, 1, 1, 1, c*2)
    scale_params = scale_params.view(
      scale_params.shape[0], *([1] * (len(x.shape) - 2)), -1
      )
    scale, bias = torch.chunk(scale_params, 2, dim=-1)
    return reshape_jax_torch(reshape_jax_torch(x) * (scale + 1) + bias)