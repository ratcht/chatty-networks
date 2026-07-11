import torch as t
import torch.nn as nn
from dataclasses import dataclass
from typing import Annotated

@dataclass(slots=True)
class Dimensions:
  state_dim: int
  n_actions: int

  key_dim: int
  value_dim: int
  gru_input_dim: int
  gru_hidden_dim: int


class Agent(nn.Module):
  def __init__(self, dims: Dimensions):
    super().__init__()
    self.dims = dims

    self.state_encoder = nn.Linear(dims.state_dim, dims.gru_input_dim)
    self.gru = nn.GRU(dims.gru_input_dim, dims.gru_hidden_dim, num_layers=1)
    
    self.query_head = nn.Linear(dims.gru_hidden_dim, dims.key_dim)
    self.key_head = nn.Linear(dims.gru_hidden_dim, dims.key_dim)
    self.value_head = nn.Linear(dims.gru_hidden_dim, dims.value_dim)

    self.action_head = nn.Linear(dims.gru_hidden_dim, dims.n_actions)
  
  def forward(
    self,
    w_t: Annotated[t.Tensor, "B state_dim"],
    c_t: Annotated[t.Tensor, "B gru_input_dim"],
    h_prev: Annotated[t.Tensor, "1, B, gru_hidden_dim"]
  ) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    policy_input = self.state_encoder(w_t) + c_t
    policy_input = policy_input.unsqueeze(0)  # (n_agents, gru_input_dim) -> (1, n_agents, gru_input_dim)

    _, h_new = self.gru(policy_input, h_prev)

    h = h_new.squeeze(0)  # (1, n_agents, gru_hidden_dim) -> (n_agents, gru_hidden_dim)

    Q_t = self.query_head(h)
    K_t = self.key_head(h)
    V_t = self.value_head(h)
    action_logits = self.action_head(h)

    return Q_t, K_t, V_t, action_logits, h