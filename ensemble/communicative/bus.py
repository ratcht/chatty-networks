import math

import torch as t
import torch.nn as nn


class QKVEncoder(nn.Module):
  """Per-specialist TarMAC heads: query (receiver), key/signature and value (sender)."""
  def __init__(self, in_dim: int, key_dim: int, value_dim: int, reduce: nn.Module):
    super().__init__()
    self.reduce = reduce
    self.query_head = nn.Linear(in_dim, key_dim)
    self.key_head = nn.Linear(in_dim, key_dim)
    self.value_head = nn.Linear(in_dim, value_dim)

  def forward(self, x) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    f = self.reduce(x)
    return self.query_head(f), self.key_head(f), self.value_head(f)


class MLPEncoder(nn.Module):
  """QKVEncoder with a shared nonlinear trunk before the three heads diverge,
  instead of three independent linear maps off the raw pooled features —
  tests whether the encoder's expressiveness, not just its output
  dimensionality, is what limits communication."""
  def __init__(self, in_dim: int, key_dim: int, value_dim: int, reduce: nn.Module,
               hidden_dim: int | None = None):
    super().__init__()
    self.reduce = reduce
    hidden_dim = hidden_dim or in_dim
    self.trunk = nn.Sequential(
      nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
    )
    self.query_head = nn.Linear(hidden_dim, key_dim)
    self.key_head = nn.Linear(hidden_dim, key_dim)
    self.value_head = nn.Linear(hidden_dim, value_dim)

  def forward(self, x) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    f = self.trunk(self.reduce(x))
    return self.query_head(f), self.key_head(f), self.value_head(f)


class MLPNoLNEncoder(nn.Module):
  """MLPEncoder with the trunk's LayerNorm removed — isolates whether the
  nonlinearity itself (vs. the normalization riding along with it) is what
  the expressiveness gain depends on."""
  def __init__(self, in_dim: int, key_dim: int, value_dim: int, reduce: nn.Module,
               hidden_dim: int | None = None):
    super().__init__()
    self.reduce = reduce
    hidden_dim = hidden_dim or in_dim
    self.trunk = nn.Sequential(
      nn.Linear(in_dim, hidden_dim), nn.GELU(),
    )
    self.query_head = nn.Linear(hidden_dim, key_dim)
    self.key_head = nn.Linear(hidden_dim, key_dim)
    self.value_head = nn.Linear(hidden_dim, value_dim)

  def forward(self, x) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    f = self.trunk(self.reduce(x))
    return self.query_head(f), self.key_head(f), self.value_head(f)


class Decoder(nn.Module):
  def __init__(self, value_dim: int, out_dim: int, expand: nn.Module):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(value_dim, value_dim),
      nn.GELU(),
      nn.Linear(value_dim, out_dim),
    )
    self.expand = expand

    last_linear = self.mlp[-1]
    assert isinstance(last_linear, nn.Linear)

    nn.init.zeros_(last_linear.weight)
    nn.init.zeros_(last_linear.bias)

  def forward(self, x):
    return self.expand(self.mlp(x))


class TarMACBus(nn.Module):
  """Bare single-head scaled dot-product attention across ensemble members.

  Each member attends over all members including itself — with n=2, excluding
  self would collapse the softmax to a constant 1.
  """
  def forward(
    self,
    Q: t.Tensor,  # (b, n, key_dim)
    K: t.Tensor,  # (b, n, key_dim)
    V: t.Tensor,  # (b, n, value_dim)
  ) -> tuple[t.Tensor, t.Tensor]:
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])  # (b, n, n)
    attn = scores.softmax(dim=-1)
    return attn @ V, attn  # (b, n, value_dim), (b, n, n)
