from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from aim import Run


@dataclass
class TrainConfig:
  epochs: int
  lr: float
  weight_decay: float
  seed: int
  log_every: int = 50
  device: str = "cuda"


@dataclass
class TrainJob:
  model: nn.Module
  loader: DataLoader
  optimizer: Optimizer
  criterion: nn.Module
  config: TrainConfig
  run: Optional[Run] = None


def train(job: TrainJob) -> list[float]:
  cfg = job.config
  job.model.to(cfg.device)
  job.model.train()
  losses: list[float] = []
  step = 0
  try:
    for epoch in range(cfg.epochs):
      for x, y in job.loader:
        x, y = x.to(cfg.device), y.to(cfg.device)
        job.optimizer.zero_grad()
        logits = job.model(x)
        loss = job.criterion(logits, y)
        loss.backward()
        job.optimizer.step()
        losses.append(loss.item())
        if job.run is not None:
          job.run.track(loss.item(), name="train_loss", step=step, context={"epoch": epoch})
        if step % cfg.log_every == 0:
          print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
        step += 1
  except KeyboardInterrupt:
    print(f"interrupted at step {step}")
  return losses