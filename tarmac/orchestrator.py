import torch as t
import torch.nn as nn
import torch.nn.functional as F
from typing import Annotated
import math
import einops

from agent import Dimensions, Agent

class TarMAC(nn.Module):
  def __init__(self, n_agents: int, agent_dimensions: Dimensions):
    super().__init__()
    self.n_agents = n_agents
    self.dims = agent_dimensions
    self.shared_agent = Agent(self.dims)

  def initial_state(self, n_envs: int = 1):
    """Zero comm + hidden state for the start of an episode."""
    c0 = t.zeros(n_envs, self.n_agents, self.dims.value_dim)
    h0 = t.zeros(1, n_envs*self.n_agents, self.dims.gru_hidden_dim)
    return c0, h0

  def forward(
    self,
    observations: Annotated[t.Tensor, "n_envs n_agents state_dim"],
    c_prev: Annotated[t.Tensor, "n_envs n_agents value_dim"],
    h_prev: Annotated[t.Tensor, "1 n_envs*n_agents gru_hidden_dim"],
  ):
    n_envs = observations.shape[0]

    Q, K, V, action_logits, H = self.shared_agent(
      einops.rearrange(observations, "e a d -> (e a) d"),
      einops.rearrange(c_prev, "e a d -> (e a) d"),
      h_prev,
    )

    Q = einops.rearrange(Q, "(e a) d -> e a d", e=n_envs)
    K = einops.rearrange(K, "(e a) d -> e a d", e=n_envs)
    V = einops.rearrange(V, "(e a) d -> e a d", e=n_envs)
    action_logits = einops.rearrange(action_logits, "(e a) d -> e a d", e=n_envs)
    hidden = einops.rearrange(H, "(e a) d -> e a d", e=n_envs)

    scores = Q @ K.transpose(-2, -1)     # (n_envs, n_agents, n_agents)
    scores = scores / math.sqrt(self.dims.key_dim)
    attn = F.softmax(scores, dim=-1)

    c_next = attn @ V                     # (n_envs, n_agents, value_dim)

    return action_logits, hidden, c_next, H.unsqueeze(0)