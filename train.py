from dataclasses import dataclass
from typing import Optional

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from aim import Run
from tqdm.auto import tqdm

@dataclass
class TrainConfig:
  epochs: int
  lr: float
  weight_decay: float
  seed: int
  dataset_split_seed: int = 0
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



def train(job: TrainJob, *args, aux_weight: float = 0.0, **kwargs) -> list[float]:
  cfg = job.config
  job.model.to(cfg.device)
  job.model.train()
  losses: list[float] = []
  step = 0
  pbar = tqdm(total=cfg.epochs * len(job.loader), desc="train")
  try:
    for epoch in range(cfg.epochs):
      for x, y in job.loader:
        x, y = x.to(cfg.device), y.to(cfg.device)
        job.optimizer.zero_grad()

        if aux_weight > 0.0:
          logits, aux_logits = job.model(x, *args, return_aux=True, **kwargs)
          main_loss = job.criterion(logits, y)
          n, b, c = aux_logits.shape
          aux_loss = job.criterion(aux_logits.view(n * b, c), y.repeat(n))
          loss = main_loss + aux_weight * aux_loss
        else:
          logits = job.model(x, *args, **kwargs)
          loss = job.criterion(logits, y)

        loss.backward()
        job.optimizer.step()
        losses.append(loss.item())
        if job.run is not None:
          job.run.track(loss.item(), name="train_loss", step=step, context={"epoch": epoch})
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        step += 1
  except KeyboardInterrupt:
    print(f"interrupted at step {step}")
  return losses