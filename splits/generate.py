"""
Generate the canonical 3-way split of the CIFAR-100 train set.

The 50k training examples are partitioned into:
  backbone_indices  — backbone training pool
  ensemble_indices  — joint ensemble training (gating / communicative)
  val_indices       — single shared validation set

The official 10k CIFAR-100 test set is never touched here; it is the test set
for all downstream evaluation.

Example
-------
uv run python splits/generate.py --seed 0
uv run python splits/generate.py --seed 1 --fracs 0.7 0.2 0.1
"""

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets

# anchor defaults to the repo root so the script works from any CWD
_REPO_ROOT = Path(__file__).resolve().parent.parent


def make_three_way_split(
    targets: np.ndarray,
    fracs: tuple[float, float, float],
    seed: int,
) -> dict[str, list[int]]:
    backbone_frac, ensemble_frac, _ = fracs
    rng = np.random.default_rng(seed)

    backbone_indices: list[int] = []
    ensemble_indices: list[int] = []
    val_indices: list[int] = []

    for cls in np.unique(targets):
        cls_idx = np.flatnonzero(targets == cls).copy()
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_backbone = int(round(backbone_frac * n))
        n_ensemble = int(round(ensemble_frac * n))
        backbone_indices.extend(cls_idx[:n_backbone].tolist())
        ensemble_indices.extend(cls_idx[n_backbone:n_backbone + n_ensemble].tolist())
        val_indices.extend(cls_idx[n_backbone + n_ensemble:].tolist())

    return {
        "backbone_indices": backbone_indices,
        "ensemble_indices": ensemble_indices,
        "val_indices": val_indices,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the 3-way CIFAR-100 train split")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fracs", type=float, nargs=3, default=[0.5, 0.4, 0.1],
                        metavar=("BACKBONE", "ENSEMBLE", "VAL"),
                        help="Fractions per class (default: %(default)s)")
    parser.add_argument("--data-root", type=str, default=str(_REPO_ROOT / "data"))
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (default: splits/three_way_seed{seed}.pt)")
    args = parser.parse_args()

    fracs = tuple(args.fracs)
    if abs(sum(fracs) - 1.0) > 1e-9:
        parser.error(f"fracs must sum to 1, got {fracs}")

    targets = np.array(datasets.CIFAR100(args.data_root, train=True, download=True).targets)
    split = make_three_way_split(targets, fracs, args.seed)

    sizes = {k: len(v) for k, v in split.items()}
    overlap = (
      set(split["backbone_indices"]) & set(split["ensemble_indices"]),
      set(split["backbone_indices"]) & set(split["val_indices"]),
      set(split["ensemble_indices"]) & set(split["val_indices"]),
    )
    assert all(len(o) == 0 for o in overlap), "subsets must be disjoint"
    assert sum(sizes.values()) == len(targets), "subsets must cover the train set"

    out = args.output or _REPO_ROOT / "splits" / f"three_way_seed{args.seed}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **split,
            "seed": args.seed,
            "fracs": {"backbone": fracs[0], "ensemble": fracs[1], "val": fracs[2]},
            "source": "cifar100_train",
            "test_set": "cifar100_test (official 10k, untouched)",
            "created": date.today().isoformat(),
        },
        out,
    )
    print(f"saved → {out}  sizes={sizes}")


if __name__ == "__main__":
    main()
