import torch
import torch.nn as nn

Tensor = torch.Tensor

def reshape_jax_torch(tensor: Tensor) -> Tensor:
    """
    A jax based dataloader is off shape (bs, width, height, depth, c),
    while a PyTorch based dataloader is off shape (bs, c, depth, height, width).

    It transforms a tensor for the 2D and 3D case as follows:
    - 2D: (bs, c, depth, height, width) <-> (bs, width, height, depth, c)
    - 3D: (bs, c, height, width) <-> (bs, width, height, c)
    """
    if len(tensor.shape) == 4:
        # Reshape for the 2D case
        return tensor.permute(0, 3, 2, 1)
    elif len(tensor.shape) == 5:
        # Reshape for the 3D case
        return tensor.permute(0, 4, 3, 2, 1)
    else:
        raise ValueError(
            f"Incorrect tensor shape, only 4D and 5D tensors are valid"
            )

def default_init(scale: float = 1e-10):
  """Initialization of weights with scaling"""
  def initializer(tensor: Tensor):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = torch.sqrt(torch.tensor(scale / ((fan_in + fan_out) / 2.0)))
    bound = torch.sqrt(torch.tensor(3.0)) * std # uniform dist. scaling factor
    with torch.no_grad():
      return tensor.uniform_(-bound, bound)
  return initializer