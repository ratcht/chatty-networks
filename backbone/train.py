"""
Train N ResNet20 backbones on the backbone split of the canonical 3-way split
(made by splits/generate.py). Two generation methods:

stratified - N stratified subsets of the backbone pool with controlled pairwise
             overlap, one training run per member with its own seed.
snapshot -  snapshot ensembling (Huang et al. 2017): one training run with a
            cyclic cosine learning rate, saving a snapshot at each cycle end.

Examples
--------
uv run python backbone/train.py stratified --seeds 42 137 \\
    --frac 0.75 --overlap-frac 0.5

uv run python backbone/train.py snapshot --seed 42 \\
    --n-snapshots 10 --cycle-epochs 20
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from backbone.resnet20 import ResNet20

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def make_seeds(seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    return {
        "init": int(rng.integers(0, 2**31)),
        "loader": int(rng.integers(0, 2**31)),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_split(split_file: Path) -> tuple[list[int], list[int]]:
    """Load the canonical 3-way split (made by splits/generate.py).

    Returns the backbone training pool and the shared validation indices.
    """
    split = torch.load(split_file, map_location="cpu", weights_only=True)
    return split["backbone_indices"], split["val_indices"]


def make_cifar100_loaders(
    train_indices: list[int],
    val_indices: list[int],
    batch_size: int,
    loader_seed: int,
    data_root: str = "data",
) -> tuple[DataLoader, DataLoader]:
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
    ])

    # Two dataset instances so val examples never see augmentation
    train_set = datasets.CIFAR100(data_root, train=True, download=True, transform=train_tf)
    val_set = datasets.CIFAR100(data_root, train=True, download=True, transform=val_tf)

    g = torch.Generator()
    g.manual_seed(loader_seed)

    train_loader = DataLoader(
        Subset(train_set, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=g,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        Subset(val_set, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    model.train()
    return correct / total


def _save(model: nn.Module, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
    print(f"  saved → {path}")


def train_backbone(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: str,
    lr: float = 0.1,
) -> nn.Module:
    model = model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        avg_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc = _evaluate(model, val_loader, device)
            print(f"  epoch {epoch+1:>3}/{epochs}  loss {avg_loss:.4f}  val_acc {acc:.3f}")

    return model


# ---------------------------------------------------------------------------
# stratified subsets + different seeds
# ---------------------------------------------------------------------------

def make_stratified_subsets(
    targets,
    n_subsets: int,
    frac: float = 0.75,
    overlap_frac: float = 0.60,
    seed: int = 0,
) -> list[list[int]]:
    """
    Create N stratified subsets with controlled pairwise per-class overlap.

    Per class, a shared "core" of overlap_frac examples is given to every
    subset, plus a disjoint unique chunk of (frac - overlap_frac) examples per
    subset — so the overlap between any pair of subsets is exactly
    overlap_frac of each class.

    Args:
        targets: list or array of integer class labels.
        n_subsets: number of subsets (ensemble members).
        frac: fraction of each class assigned to each subset.
        overlap_frac: fraction of each class shared by all subsets.
        seed: RNG seed.

    Returns:
        list of n_subsets index lists.

    Constraints:
        overlap_frac <= frac
        overlap_frac >= (n_subsets * frac - 1) / (n_subsets - 1)

    The second condition ensures each class has enough examples for the core
    plus n_subsets disjoint unique chunks. For n_subsets=2 it reduces to
    overlap_frac >= 2 * frac - 1.
    """
    if n_subsets < 2:
        raise ValueError("n_subsets must be at least 2.")

    if overlap_frac > frac:
        raise ValueError("overlap_frac cannot exceed frac.")

    min_overlap = max(0.0, (n_subsets * frac - 1.0) / (n_subsets - 1.0))
    if overlap_frac < min_overlap:
        raise ValueError(
            f"overlap_frac is too small for {n_subsets} subsets of frac={frac}. "
            f"Need overlap_frac >= {min_overlap:.3f}."
        )

    targets = np.asarray(targets)
    classes = np.unique(targets)
    rng = np.random.default_rng(seed)

    subsets: list[list[int]] = [[] for _ in range(n_subsets)]

    for cls in classes:
        cls_indices = np.flatnonzero(targets == cls).copy()
        rng.shuffle(cls_indices)

        n_cls = len(cls_indices)
        n_each = int(round(frac * n_cls))
        n_overlap = int(round(overlap_frac * n_cls))
        # clamp so rounding at the feasibility boundary can't overflow the class;
        # subset sizes may come out one example short of frac * n_cls
        n_unique = min(n_each - n_overlap, (n_cls - n_overlap) // n_subsets)

        core = cls_indices[:n_overlap]
        for i in range(n_subsets):
            start = n_overlap + i * n_unique
            unique = cls_indices[start:start + n_unique]
            subsets[i].extend(np.concatenate([core, unique]).tolist())

    for s in subsets:
        rng.shuffle(s)

    return subsets


def train_stratified(args: argparse.Namespace) -> None:
    pool, val_indices = load_split(args.split_file)
    n = len(args.seeds)
    print(f"[split] {args.split_file}  backbone pool={len(pool)}  val={len(val_indices)}")

    if args.frac == 1.0:
        # every member trains on the full pool; only the seeds differ
        member_indices = [list(pool) for _ in range(n)]
        print(f"[stratified] frac=1.0 → all {n} members share the full pool")
    else:
        # Stratified subsets drawn from the backbone pool, not the full 50k
        all_targets = np.array(datasets.CIFAR100(args.data_root, train=True, download=True).targets)
        pool_targets = all_targets[pool]
        pool_subsets = make_stratified_subsets(
            pool_targets, n_subsets=n, frac=args.frac,
            overlap_frac=args.overlap_frac, seed=args.subset_seed,
        )
        # Remap pool-local indices back to absolute CIFAR-100 train indices
        member_indices = [[pool[i] for i in sub] for sub in pool_subsets]

        sets = [set(m) for m in member_indices]
        overlaps = [
            len(sets[i] & sets[j]) / len(pool)
            for i in range(n) for j in range(i + 1, n)
        ]
        print(f"[stratified] n={n}  frac={args.frac}  "
              f"pairwise overlap min={min(overlaps):.3f} max={max(overlaps):.3f} "
              f"(requested {args.overlap_frac})  |subset|={len(member_indices[0])}")

    for seed, indices in zip(args.seeds, member_indices):
        seeds = make_seeds(seed)
        print(f"\n[stratified] backbone seed={seed}  init_seed={seeds['init']}  loader_seed={seeds['loader']}")
        seed_everything(seeds["init"])
        model = ResNet20()
        train_loader, val_loader = make_cifar100_loaders(
            indices, val_indices, args.batch_size, seeds["loader"], args.data_root
        )
        train_backbone(model, train_loader, val_loader, args.epochs, args.device, lr=args.lr)
        _save(
            model,
            args.output_dir / f"stratified_backbone_seed{seed}_epoch{args.epochs}.pt",
            {
                "method": "stratified", "seed": seed, "split_file": str(args.split_file),
                "frac": args.frac, "overlap_frac": args.overlap_frac,
                "subset_seed": args.subset_seed, "epochs": args.epochs,
            },
        )


# ---------------------------------------------------------------------------
# snapshot ensembling (one run, cyclic cosine LR)
# ---------------------------------------------------------------------------

def train_snapshot(args: argparse.Namespace) -> None:
    pool, val_indices = load_split(args.split_file)
    print(f"[split] {args.split_file}  backbone pool={len(pool)}  val={len(val_indices)}")

    n = args.n_snapshots
    total_epochs = n * args.cycle_epochs
    seeds = make_seeds(args.seed)
    print(f"[snapshot] seed={args.seed}  {n} snapshots × {args.cycle_epochs} epochs "
          f"= {total_epochs} epochs  init_seed={seeds['init']}  loader_seed={seeds['loader']}")

    seed_everything(seeds["init"])
    model = ResNet20().to(args.device)
    train_loader, val_loader = make_cifar100_loaders(
        pool, val_indices, args.batch_size, seeds["loader"], args.data_root
    )

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4
    )
    # cyclic cosine: anneal to ~0 within each cycle, then warm-restart to lr
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.cycle_epochs, T_mult=1
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(total_epochs):
        lr_now = optimizer.param_groups[0]["lr"]
        avg_loss = _train_epoch(model, train_loader, optimizer, criterion, args.device)
        scheduler.step()

        cycle_end = (epoch + 1) % args.cycle_epochs == 0
        if (epoch + 1) % 10 == 0 or epoch == 0 or cycle_end:
            print(f"  epoch {epoch+1:>3}/{total_epochs}  lr {lr_now:.4f}  loss {avg_loss:.4f}")

        if cycle_end:
            k = (epoch + 1) // args.cycle_epochs
            acc = _evaluate(model, val_loader, args.device)
            print(f"  [snapshot {k}/{n}] val_acc {acc:.3f}")
            _save(
                model,
                args.output_dir / f"snapshot_backbone_seed{args.seed}_snap{k}of{n}.pt",
                {
                    "method": "snapshot", "seed": args.seed, "split_file": str(args.split_file),
                    "snapshot": k, "n_snapshots": n, "cycle_epochs": args.cycle_epochs,
                    "epoch": epoch + 1, "val_acc": acc,
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet20 backbones for CIFAR-100 ensembles")
    sub = parser.add_subparsers(dest="method", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--split-file", type=Path, default=Path("splits/three_way_seed0.pt"),
                        help="3-way split file made by splits/generate.py")
    common.add_argument("--lr", type=float, default=0.1)
    common.add_argument("--batch-size", type=int, default=128)
    common.add_argument("--output-dir", type=Path, default=Path("backbone/checkpoints"))
    common.add_argument("--data-root", type=str, default="data")
    common.add_argument("--device", type=str, default="cuda")

    p_strat = sub.add_parser("stratified", parents=[common],
                             help="N stratified subsets + different seeds")
    p_strat.add_argument("--seeds", type=int, nargs="+", required=True,
                         help="One seed per ensemble member (N = number of seeds)")
    p_strat.add_argument("--frac", type=float, default=0.75,
                         help="Fraction of each class per subset; 1.0 = full pool for everyone")
    p_strat.add_argument("--overlap-frac", type=float, default=0.5,
                         help="Fraction of each class shared by all subsets")
    p_strat.add_argument("--subset-seed", type=int, default=0,
                         help="Seed for drawing the subsets (independent of member seeds)")
    p_strat.add_argument("--epochs", type=int, default=200)

    p_snap = sub.add_parser("snapshot", parents=[common],
                            help="Snapshot ensemble: one run, cyclic cosine LR")
    p_snap.add_argument("--seed", type=int, required=True)
    p_snap.add_argument("--n-snapshots", type=int, required=True)
    p_snap.add_argument("--cycle-epochs", type=int, default=20,
                        help="Epochs per LR cycle (total = n_snapshots × cycle_epochs)")

    args = parser.parse_args()

    if args.method == "stratified":
        train_stratified(args)
    else:
        train_snapshot(args)


if __name__ == "__main__":
    main()
