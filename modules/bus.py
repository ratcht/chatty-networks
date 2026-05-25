import torch as t
import torch.nn as nn
import torch.nn.functional as F

import einops.layers.torch as et

class Encoder(nn.Module):
  def __init__(self, in_dim: int, msg_dim: int, reduce: nn.Module):
    super().__init__()
    self.reduce = reduce
    self.mlp = nn.Sequential(
      nn.Linear(in_dim, in_dim),
      nn.GELU(),
      nn.Linear(in_dim, msg_dim)
    )
  
  def forward(self, x):
    return self.mlp(self.reduce(x))
  

class Decoder(nn.Module):
  def __init__(self, msg_dim: int, out_dim: int, expand: nn.Module):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(msg_dim, msg_dim),
      nn.GELU(),
      nn.Linear(msg_dim, out_dim),
    )
    self.expand = expand
    nn.init.zeros_(self.mlp[-1].weight)
    nn.init.zeros_(self.mlp[-1].bias)

  def forward(self, x):
    return self.expand(self.mlp(x))


class Bus(nn.Module):
  def __init__(self, msg_dim: int, num_heads: int = 4, dropout: float = 0.1):
    super().__init__()
    self.norm = nn.LayerNorm(msg_dim)
    self.attn = nn.MultiheadAttention(
      msg_dim, num_heads=num_heads, dropout=dropout, batch_first=True
    )
    self.drop = nn.Dropout(dropout)

  def forward(self, x):
    h = self.norm(x)
    h, w = self.attn(query=h, key=h, value=h)
    return x + self.drop(h), w
