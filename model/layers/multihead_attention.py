import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadDotProductAttention(nn.Module):
    """Mulit Head Dot Product Attention with querry and key normalization"""
    
    def __init__(self, emb_dim: int, num_heads: int, 
                 normalize_qk: bool=False, dropout: float=0.0):
        super(MultiHeadDotProductAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.normalize_qk = normalize_qk
        self.dropout = dropout

        if emb_dim % num_heads != 0:
            raise ValueError(
                "Embedding Dimension must be divisible through the number of heads"
                )
        self.multihead_attention = nn.MultiheadAttention(
            self.emb_dim, self.num_heads, dropout=dropout
            )
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.multihead_attention.in_proj_weight)
        nn.init.xavier_uniform_(self.multihead_attention.out_proj.weight)

    def forward(self, query, key, value):
        if self.normalize_qk:
            query = F.normalize(query, p=2, dim=-1)
            key = F.normalize(key, p=2, dim=-1)

        out, _ = self.multihead_attention(query, key, value)

        return out