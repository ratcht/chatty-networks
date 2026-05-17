from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train(
  model: nn.Module,
  loader: DataLoader,
  optimizer: Optimizer,
  criterion: nn.Module,
  epochs: int,
  device: str = "cuda",
  run=None,
  log_every: int = 50,
) -> list[float]:
  model.to(device)
  model.train()
  losses: list[float] = []
  step = 0
  try:
    for epoch in range(epochs):
      for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if run is not None:
          run.track(loss.item(), name="train_loss", step=step, context={"epoch": epoch})
        if step % log_every == 0:
          print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
        step += 1
  except KeyboardInterrupt:
    print(f"interrupted at step {step}")
  return losses