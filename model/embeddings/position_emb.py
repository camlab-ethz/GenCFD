from typing import Callable, Optional
import torch
import torch.nn as nn

Tensor = torch.Tensor


def position_embedding(ndim: int, **kwargs) -> nn.Module:
    if ndim == 1:
        return Add1dPosEmbedding(**kwargs)
    elif ndim == 2:
        return Add2dPosEmbedding(**kwargs)
    elif ndim == 3:
        return Add3dPosEmbedding(**kwargs)
    else:
        raise ValueError("Only 1D, 2D, 3D position embeddings are supported.")
    

class Add1dPosEmbedding(nn.Module):
    """Adds a trainable 1D position embeddings to the inputs."""

    def __init__(self, emb_init: Callable[[Tensor, float, float], None]=nn.init.normal_, 
                 stddev: float=0.02):
        super(Add1dPosEmbedding, self).__init__()
        self.emb_init = emb_init
        self.stddev = stddev
        self.pos_emb = None

    def forward(self, x: Tensor) -> Tensor:
        # Input shape of the tensor: (bs, c, l)
        assert len(x.shape) == 3
        _, c, l = x.shape
        if self.pos_emb is None:
            self.pos_emb = nn.Parameter(torch.empty(c, l))
            self.emb_init(self.pos_emb, mean = 0.0, std = self.stddev)
        
        return x + self.pos_emb.unsqueeze(0)


class Add2dPosEmbedding(nn.Module):
    """Adds a trainable 2D position embeddings to the inputs."""

    def __init__(self, emb_init: Callable[[Tensor, float, float], None]=nn.init.normal_,
                 stddev: float=0.02):
        super(Add2dPosEmbedding, self).__init__()

        self.emb_init = emb_init
        self.stddev = stddev
        self.row_emb = None
        self.col_emb = None

    def forward(self, x: Tensor) -> Tensor:
        # Input shape of the tensor: (bs, c, h, w)
        assert len(x.shape) == 4
        _, c, h, w = x.shape
        assert c % 2 == 0, "Number of channels must be even"

        if self.row_emb is None or self.col_emb is None:
            self.row_emb = nn.Parameter(torch.empty(c // 2, w))
            self.col_emb = nn.Parameter(torch.empty(c // 2, h))
            self.emb_init(self.row_emb, mean=0.0, std=self.stddev)
            self.emb_init(self.col_emb, mean=0.0, std=self.stddev)

        row_emb = self.row_emb.unsqueeze(1).repeat(1, h, 1)  # (c, h, w)
        col_emb = self.col_emb.unsqueeze(-1).repeat(1, 1, w) # (c, h, w)

        pos_emb = torch.cat([col_emb, row_emb], dim=0) 

        return x + pos_emb.unsqueeze(0)


class Add3dPosEmbedding(nn.Module):
    """Adds a trainable 2D position embeddings to the inputs."""

    def __init__(self, emb_init: Callable[[Tensor, float, float], None]=nn.init.normal_, 
                 stddev: float=0.02):
        super(Add3dPosEmbedding, self).__init__()

        self.emb_init = emb_init
        self.stddev = stddev
        self.row_emb = None
        self.col_emb = None
        self.depth_emb = None

    def forward(self, x: Tensor) -> Tensor:
        # x.shape: (bs, c, depth, height, width)
        assert len(x.shape) == 5
        _, c, d, h, w = x.shape
        assert c % 3 == 0, "Number of channels must be divisible through 3"

        if self.row_emb is None or self.col_emb is None or self.depth_emb is None:
            self.row_emb = nn.Parameter(torch.empty(c // 3, d, w))
            self.col_emb = nn.Parameter(torch.empty(c // 3, d, h))
            self.depth_emb = nn.Parameter(torch.empty(c // 3, h, w))
            self.emb_init(self.row_emb, mean=0.0, std=self.stddev)
            self.emb_init(self.col_emb, mean=0.0, std=self.stddev)
            self.emb_init(self.depth_emb, mean=0.0, std=self.stddev)

        row_emb = self.row_emb.unsqueeze(2).repeat(1, 1, h, 1)
        col_emb = self.col_emb.unsqueeze(-1).repeat(1, 1, 1, w)
        depth_emb = self.depth_emb.unsqueeze(1).repeat(1, d, 1, 1)

        pos_emb = torch.cat([depth_emb, col_emb, row_emb], dim=0)

        return x + pos_emb