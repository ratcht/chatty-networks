"""
Generate the canonical 3-way split of the CIFAR-100 train set, plus the seed
lists the rest of the pipeline runs on.

The 50k training examples are partitioned into:
  backbone_indices  — backbone training pool
  ensemble_indices  — joint ensemble training (gating / communicative)
  val_indices       — single shared validation set

The official 10k CIFAR-100 test set is never touched here; it is the test set
for all downstream evaluation.

Alongside the partition this records two seed lists, both derived from the
single master --seed so the whole pipeline is reproducible from one number:

  expert_seeds  — one per snapshot-ensemble trajectory (stage 2, the expert
                  pool). N_1 trajectories x M snapshots = the candidate pool
                  that selection later picks a group from.
  joint_seeds   — one per replicate of a joint-ensemble config (stage 4),
                  used to average a result over seeds rather than trust one.

Example
-------
uv run python splits/generate.py --seed 0
uv run python splits/generate.py --seed 1 --fracs 0.7 0.2 0.1
uv run python splits/generate.py --seed 0 --n-expert-seeds 10 --n-joint-seeds 5
"""

import argparse
import hashlib
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


# Distinct stream ids so each seed list is drawn from its own RNG, independent
# of the partition's. Changing --n-expert-seeds can then never perturb the
# partition, and never perturbs the joint seeds either.
_EXPERT_STREAM = 1
_JOINT_STREAM = 2


def derive_seeds(master_seed: int, n: int, stream: int) -> list[int]:
    """Draw n child seeds from a stream keyed by (master_seed, stream).

    Drawn sequentially, so the list grows by appending: regenerating with a
    larger n leaves the earlier seeds untouched. That matters because those
    seeds label trajectories that may already have cost hours to train — a
    seed list that reshuffled when extended would silently orphan them.
    """
    rng = np.random.default_rng([master_seed, stream])
    return [int(s) for s in rng.integers(0, 2**31, size=n)]


def read_seeds(split_file: Path, key: str) -> tuple[list[int], str]:
    """Return a named seed list from a split file, plus a digest of the partition.

    The digest matters because this module writes in place: bumping
    --n-expert-seeds or --n-joint-seeds rewrites the same path, and so does
    changing --fracs. The path alone can't tell those apart, so a
    repartitioned split would sail past a downstream config guard undetected.

    Shared by scripts/generate_pool.py (key="expert_seeds") and
    scripts/replicate_joint.py (key="joint_seeds") — same split file, two
    independent seed streams (see _EXPERT_STREAM/_JOINT_STREAM above).
    """
    split = torch.load(split_file, map_location="cpu", weights_only=True)
    seeds = split.get(key)
    if not seeds:
        raise SystemExit(
            f"{split_file} has no {key} — it predates seed lists. "
            f"Regenerate it: uv run python splits/generate.py --seed {split.get('seed', 0)}"
        )
    h = hashlib.sha256()
    for idx_key in ("backbone_indices", "ensemble_indices", "val_indices"):
        h.update(np.asarray(split[idx_key], dtype=np.int64).tobytes())
    return [int(s) for s in seeds], h.hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the 3-way CIFAR-100 train split")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-expert-seeds", type=int, default=10,
                        help="N_1: snapshot-ensemble trajectories to generate (default: %(default)s)")
    parser.add_argument("--n-joint-seeds", type=int, default=5,
                        help="N_2: seed replicates per joint-ensemble config (default: %(default)s)")
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

    expert_seeds = derive_seeds(args.seed, args.n_expert_seeds, _EXPERT_STREAM)
    joint_seeds = derive_seeds(args.seed, args.n_joint_seeds, _JOINT_STREAM)

    out = args.output or _REPO_ROOT / "splits" / f"three_way_seed{args.seed}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **split,
            "seed": args.seed,
            "fracs": {"backbone": fracs[0], "ensemble": fracs[1], "val": fracs[2]},
            "expert_seeds": expert_seeds,
            "joint_seeds": joint_seeds,
            "source": "cifar100_train",
            "test_set": "cifar100_test (official 10k, untouched)",
            "created": date.today().isoformat(),
        },
        out,
    )
    print(f"saved → {out}  sizes={sizes}")
    print(f"  expert_seeds (N_1={len(expert_seeds)}): {expert_seeds}")
    print(f"  joint_seeds  (N_2={len(joint_seeds)}): {joint_seeds}")


if __name__ == "__main__":
    main()
