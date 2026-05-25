from contextlib import ExitStack

import torch as t
import torch.nn as nn

from driver import Driver
from bus import Encoder, Decoder, Bus
from hooks import hook_encoder, hook_decoder, HookState

from typing import Annotated

class Specialist(nn.Module):
  def __init__(self, driver: Driver, encoder: Encoder, decoder: Decoder):
    super().__init__()
    self.driver = driver
    self.encoder = encoder
    self.decoder = decoder


class Orchestrator(nn.Module):
  def __init__(self, specialists: list[Specialist], bus: Bus, msg_dim: int, num_classes: int):
    super().__init__()
    self.specialists = nn.ModuleList(specialists)
    self.bus = bus

    self.msg_dim = msg_dim
    self.num_classes = num_classes

  def train(self, mode: bool = True):
    super().train(mode)
    for s in self.specialists:
      s.driver.backbone.eval() # keep frozen backbones in eval so batchnorm stats dont drift
    return self

  def _register_hooks(self, stack: ExitStack, decoder_outs: list[HookState], encoder_ins: list[HookState]):
    for i, s in enumerate(self.specialists):
      stack.enter_context(hook_decoder(s.driver.early, decoder_outs[i]))
      stack.enter_context(hook_encoder(s.driver.late, encoder_ins[i]))

  def forward(self, xs: list[t.Tensor], k_rounds: int) -> Annotated[t.Tensor, "b c"]:
    n: Annotated[int, "num_specialists"] = len(self.specialists)
    b: Annotated[int, "batch_size"] = xs[0].shape[0]
    d: Annotated[int, "msg_dim"] = self.msg_dim
    c: Annotated[int, "num_classes"] = self.num_classes

    device = xs[0].device

    msgs = t.zeros((b, n, d), device=device)

    decoder_outs: list[HookState] = [HookState() for _ in range(n)]
    encoder_ins: list[HookState] = [HookState() for _ in range(n)]
    outputs = t.zeros((n, b, c), device=device)

    with ExitStack() as stack:
      # register encoder/decoder hooks
      self._register_hooks(stack, decoder_outs, encoder_ins)
      
      # run k rounds
      for _ in range(k_rounds):
        encoder_outs = t.zeros((b, n, d), device=device)

        for i, s in enumerate(self.specialists):
          # decode messages
          decoder_outs[i].value = s.decoder(msgs[:, i])

          # run backbone
          outputs[i] = s.driver.backbone(xs[i])
          
          # encode outgoing messages
          encoder_outs[:, i] = s.encoder(encoder_ins[i].value)

        # run bus
        msgs, _ = self.bus(encoder_outs)

    logits: Annotated[t.Tensor, "b c"] = outputs.mean(dim=0)
    return logits