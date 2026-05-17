from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
  model: nn.Module,
  loader: DataLoader,
  device: str = "cuda",
  criterion: Optional[nn.Module] = None,
) -> dict[str, float]:
  model.to(device)
  model.eval()
  correct = 0
  total = 0
  total_loss = 0.0
  for x, y in loader:
    x, y = x.to(device), y.to(device)
    logits = model(x)
    pred = logits.argmax(dim=-1)
    correct += (pred == y).sum().item()
    total += y.size(0)
    if criterion is not None:
      total_loss += criterion(logits, y).item() * y.size(0)
  out = {"accuracy": correct / total}
  if criterion is not None:
    out["loss"] = total_loss / total
  return out


def plot_loss(
  losses: list[float],
  smooth: int = 50,
  save_path: Optional[str] = None,
) -> None:
  fig, ax = plt.subplots(figsize=(8, 4))
  ax.plot(losses, alpha=0.3, label="raw")
  if smooth > 1 and len(losses) > smooth:
    smoothed = [
      sum(losses[max(0, i - smooth):i + 1]) / min(i + 1, smooth)
      for i in range(len(losses))
    ]
    ax.plot(smoothed, label=f"smoothed (w={smooth})")
  ax.set_xlabel("step")
  ax.set_ylabel("loss")
  ax.legend()
  plt.tight_layout()
  if save_path:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
  plt.show()


def plot_metrics(
  metrics: dict[str, float],
  save_path: Optional[str] = None,
) -> None:
  keys = list(metrics.keys())
  values = [metrics[k] for k in keys]
  fig, ax = plt.subplots(figsize=(6, 4))
  bars = ax.bar(keys, values)
  for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom")
  plt.tight_layout()
  if save_path:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
  plt.show()