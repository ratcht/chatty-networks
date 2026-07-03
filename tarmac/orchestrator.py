import torch as t
import torch.nn as nn
import torch.nn.functional as F
from typing import Annotated
import math

from agent import Dimensions, Agent

class TarMAC(nn.Module):
  def __init__(self, n_agents: int, agent_dimensions: Dimensions):
    super().__init__()
    self.n_agents = n_agents
    self.dims = agent_dimensions
    self.shared_agent = Agent(self.dims)

  def initial_state(self):
    """Zero comm + hidden state for the start of an episode."""
    c0 = t.zeros(self.n_agents, self.dims.value_dim)
    h0 = t.zeros(self.n_agents, 1, self.dims.gru_hidden_dim)
    return c0, h0

  def forward(
    self,
    observations: Annotated[t.Tensor, "n_agents state_dim"],
    c_prev: Annotated[t.Tensor, "n_agents value_dim"],
    h_prev: Annotated[t.Tensor, "n_agents 1 gru_hidden_dim"],
  ):
    Q, K, V, action_logits, H = [], [], [], [], []

    for i in range(self.n_agents):
      q, k, v, a, h = self.shared_agent(observations[i], c_prev[i], h_prev[i])
      Q.append(q)
      K.append(k)
      V.append(v)
      action_logits.append(a)
      H.append(h)

    Q = t.stack(Q)                       # (n_agents, key_dim)
    K = t.stack(K)                       # (n_agents, key_dim)
    V = t.stack(V)                       # (n_agents, value_dim)
    action_logits = t.stack(action_logits)  # (n_agents, n_actions)
    h_next = t.stack(H)                   # (n_agents, 1, gru_hidden_dim)

    scores = Q @ K.transpose(-2, -1)     # (n_agents, n_agents)
    scores = scores / math.sqrt(self.dims.key_dim)
    attn = F.softmax(scores, dim=-1)

    c_next = attn @ V                     # (n_agents, value_dim)

    return action_logits, c_next, h_next