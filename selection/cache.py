"""
Data plumbing for expert selection: discovering the pool, caching per-expert
predictions, and reading/writing metric columns.

Two cache layers, because they go stale for different reasons.

Layer 1, per-expert predictions, is the only part that touches the GPU. It is
keyed by the checkpoint's content hash and the split's content hash, so it
survives the pool growing: new experts add files and existing ones are reused
untouched.

Layer 2, metric columns, is keyed by the whole pool plus the split plus k, and
is stored one file per metric. Columnar matters because a single table would
mean that adding a seventh metric, or fixing a bug in one, invalidates all of
them; separate columns let a metric's version bump invalidate only itself.

Everything is keyed on *content*, never on paths. `backbone/checkpoints/pool/`
is the same string before and after a trajectory drops five new checkpoints
into it, so a path-keyed cache would happily serve rankings computed over a
ten-expert pool while you believed you were looking at fifty.
"""

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from backbone.resnet import ResNet
from backbone.train import _CIFAR100_MEAN, _CIFAR100_STD

_META_VERSION = 1

# Batch 192 hangs and 256 aborts with HSA_STATUS_ERROR_INVALID_ISA inside
# MIOpen on this machine's gfx1102; 64 and 128 are the verified shapes.
_EVAL_BATCH = 128


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# digests
# ---------------------------------------------------------------------------

def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def split_digest(split: dict) -> str:
    """Content hash of the partition.

    Deliberately hashes the same three index arrays in the same order as
    scripts/generate_pool.py:read_split, so a pool and the selection run over
    it show the same digest string and can be matched by eye.
    """
    h = hashlib.sha256()
    for key in ("backbone_indices", "ensemble_indices", "val_indices"):
        h.update(np.asarray(split[key], dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def read_split(split_file: Path) -> tuple[list[int], str]:
    """Return the validation indices and the partition's digest."""
    split = torch.load(split_file, map_location="cpu", weights_only=True)
    return [int(i) for i in split["val_indices"]], split_digest(split)


# ---------------------------------------------------------------------------
# the expert pool
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Expert:
    index: int
    path: str
    digest: str
    seed: int
    snapshot: int
    depth: int
    val_acc: float

    @property
    def label(self) -> str:
        return f"s{self.seed}/snap{self.snapshot}"


def discover_experts(pool_dir: Path) -> list[Expert]:
    """Find the pool's checkpoints and put them in canonical order.

    Ordered by (seed, snapshot) from the checkpoint metadata, with the content
    digest as a tiebreak. Never by glob order, which is filesystem-dependent —
    row i of a metric column has to mean the same combination on every machine
    and after every rewrite of the directory.
    """
    paths = sorted(pool_dir.glob("*.pt"))
    if not paths:
        raise SystemExit(f"no checkpoints found in {pool_dir}")

    found = []
    for p in paths:
        meta = torch.load(p, map_location="cpu", weights_only=True).get("metadata", {})
        found.append((
            int(meta.get("seed", -1)),
            int(meta.get("snapshot", -1)),
            file_digest(p),
            p,
            int(meta.get("depth", 20)),
            float(meta.get("val_acc", float("nan"))),
        ))
    found.sort(key=lambda t: (t[0], t[1], t[2]))

    return [
        Expert(index=i, path=str(p), digest=dig, seed=seed,
               snapshot=snap, depth=depth, val_acc=acc)
        for i, (seed, snap, dig, p, depth, acc) in enumerate(found)
    ]


def pool_digest(experts: list[Expert]) -> str:
    h = hashlib.sha256()
    for e in experts:
        h.update(e.digest.encode())
    return h.hexdigest()[:16]


def load_expert(path: str, device: str) -> ResNet:
    """Mirrors ensemble/train.py:_load_backbone, kept local so selection does
    not import ensemble.train and drag in aim and the orchestrators."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    depth = ckpt.get("metadata", {}).get("depth", 20)
    model = ResNet(depth=depth).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# layer 1: per-expert predictions on the validation split
# ---------------------------------------------------------------------------

def make_val_loader(val_indices: list[int], data_root: str) -> DataLoader:
    """Validation loader with eval transforms only — no augmentation, no
    shuffling, so an expert's predictions are a pure function of its weights."""
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
    ])
    val_set = datasets.CIFAR100(data_root, train=True, download=True, transform=eval_tf)
    return DataLoader(
        Subset(val_set, val_indices),
        batch_size=_EVAL_BATCH,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


def _save_npy(path: Path, array: np.ndarray) -> None:
    """Atomic, so a kill mid-write cannot leave a file that exists but will not
    load — which a resume would otherwise skip over as if it were complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # written through a handle, not a path: np.save silently appends ".npy" to
    # any filename that lacks it, which would leave the temp file somewhere the
    # rename below never looks
    with tmp.open("wb") as f:
        np.save(f, array)
    os.replace(tmp, path)


@torch.no_grad()
def _predict(model: ResNet, loader: DataLoader, device: str) -> np.ndarray:
    probs = []
    for x, _ in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
    return torch.cat(probs).numpy().astype(np.float32)


def ensure_predictions(
    experts: list[Expert],
    val_indices: list[int],
    split_dig: str,
    cache_root: Path,
    device: str,
    data_root: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs of shape (N, n_val, C), labels of shape (n_val,)).

    Runs inference only for experts whose predictions are not already cached.
    """
    pred_dir = cache_root / "preds" / split_dig
    labels_path = pred_dir / "labels.npy"

    loader = None
    missing = [e for e in experts if not (pred_dir / f"{e.digest}.npy").exists()]
    if missing or not labels_path.exists():
        loader = make_val_loader(val_indices, data_root)

    if not labels_path.exists():
        labels = torch.cat([y for _, y in loader]).numpy().astype(np.int64)
        _save_npy(labels_path, labels)
    labels = np.load(labels_path)

    if missing:
        print(f"[select] running inference for {len(missing)} of {len(experts)} experts")
        for n, e in enumerate(missing, 1):
            model = load_expert(e.path, device)
            probs = _predict(model, loader, device)
            _save_npy(pred_dir / f"{e.digest}.npy", probs)
            acc = (probs.argmax(1) == labels).mean()
            print(f"  [{n:>3}/{len(missing)}] {e.label:<22} val_acc {acc:.4f}")
            del model
    else:
        print(f"[select] predictions cached for all {len(experts)} experts")

    probs = np.stack([np.load(pred_dir / f"{e.digest}.npy") for e in experts])
    return probs, labels


# ---------------------------------------------------------------------------
# layer 2: metric columns
# ---------------------------------------------------------------------------

def columns_dir(cache_root: Path, pool_dig: str, split_dig: str, k: int) -> Path:
    return cache_root / "columns" / f"{pool_dig}_{split_dig}_k{k}"


def load_meta(cdir: Path) -> dict:
    path = cdir / "_meta.json"
    if not path.exists():
        return {"version": _META_VERSION, "metrics": {}}
    with path.open() as f:
        return json.load(f)


def save_meta(cdir: Path, meta: dict) -> None:
    cdir.mkdir(parents=True, exist_ok=True)
    tmp = cdir / "_meta.json.tmp"
    with tmp.open("w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, cdir / "_meta.json")


def save_experts(cdir: Path, experts: list[Expert]) -> None:
    cdir.mkdir(parents=True, exist_ok=True)
    tmp = cdir / "experts.json.tmp"
    with tmp.open("w") as f:
        json.dump([asdict(e) for e in experts], f, indent=2)
    os.replace(tmp, cdir / "experts.json")


def load_experts(cdir: Path) -> list[Expert]:
    with (cdir / "experts.json").open() as f:
        return [Expert(**d) for d in json.load(f)]


def save_column(cdir: Path, metric: str, column: np.ndarray) -> None:
    _save_npy(cdir / f"{metric}.npy", column)


def load_column(cdir: Path, metric: str) -> np.ndarray:
    path = cdir / f"{metric}.npy"
    if not path.exists():
        raise SystemExit(
            f"metric {metric!r} has not been computed for this pool/split/k.\n"
            f"  expected {path}\n"
            f"  run: uv run python selection/select.py compute --k <k> --metrics {metric}"
        )
    return np.load(path)


def cached_metrics(cdir: Path) -> list[str]:
    return sorted(p.stem for p in cdir.glob("*.npy"))


# ---------------------------------------------------------------------------
# combination indexing
# ---------------------------------------------------------------------------

def combination_at(index: int, n: int, k: int) -> tuple[int, ...]:
    """The index-th k-combination of range(n) in itertools.combinations order.

    Unranked directly rather than by iterating, so pulling row 2,000,000 out of
    a k=5 column is instant.
    """
    out: list[int] = []
    remaining = index
    start = 0
    for i in range(k):
        for v in range(start, n):
            count = math.comb(n - v - 1, k - i - 1)
            if remaining < count:
                out.append(v)
                start = v + 1
                break
            remaining -= count
        else:
            raise IndexError(f"index {index} out of range for C({n},{k})")
    return tuple(out)
