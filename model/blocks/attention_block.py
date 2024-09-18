import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.residual import CombineResidualWithSkip
from model.layers.multihead_attention import MultiHeadDotProductAttention

Tensor = torch.Tensor

class AttentionBlock(nn.Module):
    """Attention block."""

    def __init__(self, num_heads: int = 1,
                normalize_qk: bool = False, dtype: torch.dtype = torch.float32):
        super(AttentionBlock, self).__init__()

        self.num_heads = num_heads
        self.normalize_qk = normalize_qk
        self.dtype = dtype

        self.norm = None
        self.multihead_attention = None

        self.res_layer = CombineResidualWithSkip()
    
    def forward(self, x: Tensor, is_training: bool) -> Tensor:
        # Input x -> (bs, widht*height, c)
        if self.norm is None:
            self.norm = nn.GroupNorm(min(x.shape[-1] // 4, 32), x.shape[-1])
        if self.multihead_attention is None:
            self.multihead_attention = MultiHeadDotProductAttention(
                x.shape[-1], self.num_heads, dropout=0.1 if is_training else 0.0
                )

        h = x.clone()
        # GroupNorm requires x -> (bs, c, widht*height)
        h = self.norm(h.permute(0, 2, 1))
        h = h.permute(0, 2, 1)
        h = self.multihead_attention(h, h, h)
        h = self.res_layer(residual=h, skip=x)

        return h


