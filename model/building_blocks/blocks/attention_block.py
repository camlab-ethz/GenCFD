import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

from model.building_blocks.layers.residual import CombineResidualWithSkip
from model.building_blocks.layers.multihead_attention import MultiHeadDotProductAttention

Tensor = torch.Tensor

class AttentionBlock(nn.Module):
    """Attention block."""

    def __init__(self, 
                 rng: torch.Generator,
                 num_heads: int = 1,
                 normalize_qk: bool = False, 
                 dtype: torch.dtype = torch.float32,
                 device: Any | None = None):
        super(AttentionBlock, self).__init__()

        self.num_heads = num_heads
        self.normalize_qk = normalize_qk
        self.dtype = dtype
        self.device = device
        self.rng = rng

        self.norm = None
        self.multihead_attention = None

        self.res_layer = CombineResidualWithSkip(
            rng=self.rng,
            dtype=self.dtype, 
            device=self.device
            )
    
    def forward(self, x: Tensor, is_training: bool) -> Tensor:
        # Input x -> (bs, widht*height, c)
        if self.norm is None:
            self.norm = nn.GroupNorm(
                min(max(x.shape[-1] // 4, 1), 32), x.shape[-1],
                device=self.device,
                dtype=self.dtype
                )
        if self.multihead_attention is None:
            self.multihead_attention = MultiHeadDotProductAttention(
                emb_dim=x.shape[-1], 
                num_heads=self.num_heads, 
                rng=self.rng,
                dropout=0.1 if is_training else 0.0,
                device=self.device, 
                dtype=self.dtype
                )

        h = x.clone()
        # GroupNorm requires x -> (bs, c, widht*height)
        h = self.norm(h.permute(0, 2, 1))
        h = h.permute(0, 2, 1) # (bs, width*height, c)
        h = self.multihead_attention(h, h, h)
        h = self.res_layer(residual=h, skip=x)

        return h


