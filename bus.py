import torch as t
import torch.nn as nn
import torch.nn.functional as F

import einops.layers.torch as et

class Encoder(nn.Module):
  def __init__(self, in_dim: int, msg_dim: int, reduce_pattern: str):
    super().__init__()
    self.block = nn.Sequential(
      et.Reduce(),
      nn.Linear(in_dim, 4*msg_dim),
      nn.GELU(),
      nn.Linear(4*msg_dim, msg_dim)
    )
  
  def forward(self, x):
    return self.block(x)


class Bus(nn.Module):
  def __init__(self):
    pass
