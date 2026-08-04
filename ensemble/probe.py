"""
Linear classifier probes (Alain & Bengio 2016) for a trained communicative
ensemble, along the message path f -> V -> c -> msg, plus depth probes over
the bare backbones to rank read-port candidates.

Probes are fit on the ensemble split (no augmentation) and scored on the
shared val split and the official test set. Chance and random-init controls
included.

Example
-------
uv run python ensemble/probe.py \\
    --checkpoint ensemble/checkpoints/communicative_snapshot_backbone_seed42_snap3of5_snapshot_backbone_seed42_snap5of5.pt
"""

import argparse
import sys
import warnings
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from ensemble.train import make_loaders, build_communicative, _load_backbone
from ensemble.communicative.hooks import hook_encoder, HookState

# a probe that fails to converge is not a measurement
warnings.filterwarnings("error", category=ConvergenceWarning)

SPLITS = ("fit", "val", "test")
BLOCKS = [f"layer{i}.{j}" for i in (1, 2, 3) for j in range(3)]


@torch.no_grad()
def collect_channel(model: nn.Module, loader: DataLoader, device: str) -> dict[str, np.ndarray]:
  """One pass with capture; returns message-1 tensors, batch-first."""
  feats = {k: [] for k in ("f", "V", "c", "msg", "logits", "y")}
  for x, y in loader:
    cap = {}
    model(x.to(device), k_rounds=1, capture=cap)
    feats["f"].append(cap["f"][0].cpu())
    feats["V"].append(cap["V"][0].cpu())
    feats["c"].append(cap["c"][0].cpu())
    feats["msg"].append(cap["msg"][0].flatten(2).cpu())
    feats["logits"].append(cap["logits"][0].transpose(0, 1).cpu())
    feats["y"].append(y)
  return {k: torch.cat(v).numpy() for k, v in feats.items()}


@torch.no_grad()
def collect_depth(backbone: nn.Module, loader: DataLoader, device: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
  """Pooled features at every block, raw at the current read port."""
  states = {name: HookState() for name in BLOCKS}
  feats: dict[str, list] = {name: [] for name in [*BLOCKS, "layer3.0/raw"]}
  ys = []
  with ExitStack() as stack:
    for name, state in states.items():
      stack.enter_context(hook_encoder(backbone.get_submodule(name), state))
    for x, y in loader:
      backbone(x.to(device))
      for name, state in states.items():
        v = state.value
        assert v is not None
        feats[name].append(v.mean((2, 3)).cpu())
        if name == "layer3.0":
          feats["layer3.0/raw"].append(v.flatten(1).cpu())
      ys.append(y)
  return {k: torch.cat(v).numpy() for k, v in feats.items()}, torch.cat(ys).numpy()


def probe(X_fit, y_fit, evals):
  clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
  clf.fit(X_fit, y_fit)
  return [clf.score(X, y) for X, y in evals]


def chance(y_fit, y_evals):
  clf = DummyClassifier(strategy="prior")
  clf.fit(np.zeros((len(y_fit), 1)), y_fit)
  return [clf.score(np.zeros((len(y), 1)), y) for y in y_evals]


def main() -> None:
  parser = argparse.ArgumentParser(description="Linear probes along the communication channel")
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--data-root", type=str, default=str(_REPO_ROOT / "data"))
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--out", type=Path, default=_REPO_ROOT / "ensemble/probe_results.csv")
  args = parser.parse_args()

  ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
  meta = ckpt["metadata"]
  backbones = [_REPO_ROOT / p for p in meta["backbones"]]

  def build():
    return build_communicative(
      backbones, meta["early"], meta["late"],
      meta["key_dim"], meta["value_dim"], 100, args.device,
    )

  model = build()
  model.load_state_dict(ckpt["state_dict"])
  model.eval()
  # same frozen backbones, untrained heads: the random-projection control
  control = build()
  control.eval()

  loaders = dict(zip(SPLITS, make_loaders(
    _REPO_ROOT / meta["split_file"], args.batch_size, args.data_root, augment=False,
  )))

  print("[collect] channel (trained)")
  data = {s: collect_channel(model, loaders[s], args.device) for s in SPLITS}
  print("[collect] channel (random control)")
  rand = {s: collect_channel(control, loaders[s], args.device) for s in SPLITS}

  rows = []

  def record(site, member, kind, n_fit, acc_v, acc_t):
    rows.append({"site": site, "member": str(member), "kind": kind,
                 "n_fit": n_fit, "val_acc": acc_v, "test_acc": acc_t})
    print(f"{site:16} member={member!s:4} {kind:10} n_fit={n_fit:6d} val={acc_v:.4f} test={acc_t:.4f}")

  def add(site, member, kind, sets):
    (X_fit, y_fit), *evals = sets
    acc_v, acc_t = probe(X_fit, y_fit, evals)
    record(site, member, kind, len(y_fit), acc_v, acc_t)

  def triple(get):
    return [get(s) for s in SPLITS]

  acc_v, acc_t = chance(data["fit"]["y"], [data["val"]["y"], data["test"]["y"]])
  record("chance", "-", "prior", len(data["fit"]["y"]), acc_v, acc_t)

  n = data["fit"]["f"].shape[1]
  for i in range(n):
    for site in ("f", "V", "c", "msg"):
      add(site, i, "trained", triple(lambda s: (data[s][site][:, i], data[s]["y"])))

    add("f+c", i, "trained", triple(lambda s: (
      np.concatenate([data[s]["f"][:, i], data[s]["c"][:, i]], axis=1), data[s]["y"])))

    # can the incoming message fix an example this member got wrong at pass 0?
    wrong = {s: data[s]["logits"][:, i].argmax(1) != data[s]["y"] for s in SPLITS}
    add("c", i, "wrong-only", triple(lambda s: (
      data[s]["c"][:, i][wrong[s]], data[s]["y"][wrong[s]])))
    acc_v, acc_t = chance(data["fit"]["y"][wrong["fit"]],
                          [data["val"]["y"][wrong["val"]], data["test"]["y"][wrong["test"]]])
    record("chance", i, "wrong-only", int(wrong["fit"].sum()), acc_v, acc_t)

    for site in ("V", "c"):
      add(site, i, "random", triple(lambda s: (rand[s][site][:, i], rand[s]["y"])))

  for bi, path in enumerate(backbones):
    print(f"[collect] depth {path.stem}")
    backbone = _load_backbone(path, args.device)
    depth = {s: collect_depth(backbone, loaders[s], args.device) for s in SPLITS}
    for site in depth["fit"][0]:
      add(f"depth/{site}", f"bb{bi}", "backbone",
          [(depth[s][0][site], depth[s][1]) for s in SPLITS])

  df = pd.DataFrame(rows)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(args.out, index=False)
  print(f"saved → {args.out}")


if __name__ == "__main__":
  main()
