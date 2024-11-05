import torch
import torch.nn as nn

from attention import MultiHeadAttention
from mlp import MLP


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float=1e-6) -> None:
        super(RMSNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class EncoderBlock(nn.Module):
    def __init__(self, dropout: float, norm: RMSNorm, mlp: MLP, attention: MultiHeadAttention) -> None:
        super().__init__()
        self.norm = norm
        self.mlp = mlp
        self.multi_h_attention = attention
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual1 = x 
        x = self.norm(x)
        _, x = self.multi_h_attention(x)
        residual2 = residual1 + x
        x = self.norm(residual2)
        x = self.mlp(x) + residual2
        
        return x
        