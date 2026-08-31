import math
from contextlib import ExitStack

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from ensemble.communicative.driver import Driver
from ensemble.communicative.bus import QKVEncoder, Decoder, TarMACBus
from ensemble.communicative.hooks import hook_encoder, hook_decoder, HookState

from typing import Annotated

class Specialist(nn.Module):
  def __init__(self, driver: Driver, decoder: Decoder, encoder: QKVEncoder):
    super().__init__()
    self.driver = driver
    self.decoder = decoder
    self.encoder = encoder


class Orchestrator(nn.Module):
  def __init__(self, specialists: list[Specialist], key_dim: int, value_dim: int, num_classes: int):
    super().__init__()
    self.specialists = nn.ModuleList(specialists)
    self.bus = TarMACBus()

    self.key_dim = key_dim
    self.value_dim = value_dim
    self.num_classes = num_classes

    # message magnitudes from the most recent communication round (see forward);
    # stays None until a forward with k_rounds >= 1 runs
    self.last_stats: dict[str, float] | None = None
    # mean per-row entropy of the bus's attention weights, averaged across the
    # batch — same round as last_stats, None under the same conditions
    self.last_attn_entropy: float | None = None
    # mean peak attention weight max_i p_i, averaged over receivers and batch —
    # the other half of the concentration picture: entropy is a whole-row
    # summary, this is just how much mass the winner took
    self.last_attn_top1: float | None = None
    # the bus's post-softmax attention itself, (b, n, n), detached — entropy
    # says how concentrated routing is, this says who it concentrated *on*
    self.last_attn: t.Tensor | None = None
    # per-specialist logits before (perception pass) and after communication,
    # detached — lets the trainer measure how messages changed the answers
    self.last_shift: dict[str, t.Tensor] | None = None

  def train(self, mode: bool = True):
    super().train(mode)
    for s in self.specialists:
      assert isinstance(s, Specialist)
      s.driver.backbone.eval() # keep frozen backbones in eval so batchnorm stats dont drift
    return self

  def _register_hooks(self, stack: ExitStack, decoder_outs: list[HookState], encoder_ins: list[HookState]):
    for i, s in enumerate(self.specialists):
      assert isinstance(s.driver, Driver)
      stack.enter_context(hook_decoder(s.driver.early, decoder_outs[i]))
      stack.enter_context(hook_encoder(s.driver.late, encoder_ins[i]))

  def forward(self, xs: t.Tensor | list[t.Tensor], k_rounds: int = 1) -> t.Tensor:
    n: Annotated[int, "num_specialists"] = len(self.specialists)
    if isinstance(xs, t.Tensor):
      xs = [xs] * n
    b: Annotated[int, "batch_size"] = xs[0].shape[0]

    device = xs[0].device

    # decoder states start as None: the first pass is pure perception, no injection
    decoder_outs: list[HookState] = [HookState() for _ in range(n)]
    encoder_ins: list[HookState] = [HookState() for _ in range(n)]
    outputs = t.zeros((n, b, self.num_classes), device=device)

    self.last_stats = None
    self.last_attn_entropy = None
    self.last_attn_top1 = None
    self.last_attn = None
    self.last_shift = None
    pre_logits = None

    with ExitStack() as stack:
      # register encoder/decoder hooks
      self._register_hooks(stack, decoder_outs, encoder_ins)

      # perception pass + k communication rounds; Q/K/V are recomputed each
      # pass from the updated late features (recurrence via the backbone)
      for round_idx in range(k_rounds + 1):
        Q = t.zeros((b, n, self.key_dim), device=device)
        K = t.zeros((b, n, self.key_dim), device=device)
        V = t.zeros((b, n, self.value_dim), device=device)

        for i, s in enumerate(self.specialists):
          assert isinstance(s, Specialist)
          outputs[i] = s.driver.backbone(xs[i])
          Q[:, i], K[:, i], V[:, i] = s.encoder(encoder_ins[i].value)

        if round_idx == k_rounds:
          break

        if round_idx == 0:
          # opinions before any message arrives
          pre_logits = outputs.detach().clone()

        # aggregate messages and stage them for injection on the next pass
        c, attn = self.bus(Q, K, V)
        for i, s in enumerate(self.specialists):
          decoder_outs[i].value = s.decoder(c[:, i])

        # mean L2 norms of what is said, what is heard, and what is injected —
        # a collapse toward zero here means the continuous messages vanished
        with t.no_grad():
          self.last_stats = {
            "value": V.norm(dim=-1).mean().item(),
            "aggregated": c.norm(dim=-1).mean().item(),
            "injection": t.stack([
              decoder_outs[i].value.flatten(1).norm(dim=1).mean() for i in range(n)
            ]).mean().item(),
          }
          # per-row (per-receiver) entropy of the attention distribution over
          # senders, averaged across receivers and batch — low entropy means a
          # receiver is listening to ~one sender, high (up to log(n), uniform)
          # means it's spreading attention evenly
          row_entropy = -(attn * (attn + 1e-12).log()).sum(dim=-1)  # (b, n)
          self.last_attn_entropy = row_entropy.mean().item()
          # peak of each receiver's distribution, averaged the same way: 1/n is
          # uniform, 1.0 is a hard one-hot gate
          self.last_attn_top1 = attn.max(dim=-1).values.mean().item()
          self.last_attn = attn.detach()

    if pre_logits is not None:
      self.last_shift = {"pre": pre_logits, "post": outputs.detach()}

    # prob-averaged readout: log( mean_i softmax(logits_i) ), computed as a
    # logsumexp over per-member log-probs for numerical stability. Returns
    # LOG-PROBABILITIES, not logits — pair with NLLLoss, never CrossEntropyLoss
    log_probs: Annotated[t.Tensor, "b c"] = (
      t.logsumexp(F.log_softmax(outputs, dim=-1), dim=0) - math.log(n)
    )
    return log_probs
