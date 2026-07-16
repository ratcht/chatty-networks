from contextlib import contextmanager
from dataclasses import dataclass

import torch as t
import torch.nn as nn

@dataclass
class HookState:
  value: t.Tensor | None = None


@contextmanager
def hook_decoder(layer: nn.Module, state: HookState):
  # HookState is mutated across rounds; None means no message yet (perception pass)
  def fn(_, inputs):
    if state.value is None:
      return
    return (inputs[0] + state.value,)
  handle = layer.register_forward_pre_hook(fn)
  try:
    yield
  finally:
    handle.remove()


@contextmanager
def hook_encoder(layer: nn.Module, state: HookState):
  def fn(_, __, output):
    state.value = output
  handle = layer.register_forward_hook(fn)
  try:
    yield
  finally:
    handle.remove()