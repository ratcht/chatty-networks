"""
Train one snapshot-ensemble trajectory of CIFAR ResNet backbones on the
backbone split of the canonical 3-way split (made by splits/generate.py).

Snapshot ensembling (Huang et al. 2017): a single training run with a cyclic
cosine learning rate, saving a snapshot of the weights at the end of each
cycle. Each warm restart kicks the model out of its current basin, so the M
snapshots are M different local optima found along one trajectory.

Diversity across the *pool* comes from running this script once per seed —
each seed is a different initialization and a different sequence of stochastic
steps, i.e. a different trajectory through loss space. N_1 trajectories x M
snapshots gives the N_1*M candidate experts that selection later picks from.
Run the whole set with scripts/generate_pool.py rather than by hand.

Example
-------
uv run python backbone/train.py --seed 42 --n-snapshots 5 --cycle-epochs 40
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from backbone.resnet import ResNet

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
    """Build the train/val loaders.

    The train loader gets an *explicit* generator seeded from loader_seed.
    This is load-bearing, not decoration: without it, shuffling and the random
    augmentations draw from the global torch RNG, whose state at this point
    depends on how many draws model construction happened to consume — so two
    runs with the same seed but differently shaped models would silently see
    different data. Every loader in this pipeline gets its own generator.
    """
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
        Subset(
            datasets.CIFAR100(data_root, train=True, download=True, transform=val_tf),
            val_indices,
        ),
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
    """Write a checkpoint atomically.

    Straight to the final path would leave a half-written file if the process
    is killed mid-write — a file that exists (so a resume would skip it) but
    won't load. Write to a temp name, then rename, which is atomic on POSIX.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, tmp)
    os.replace(tmp, path)
    print(f"  saved → {path}")


def snapshot_filename(arch: str, seed: int, k: int, n_snapshots: int) -> str:
    """Checkpoint name for snapshot k of a trajectory. The arch prefix keeps
    pools of different backbones from colliding in one directory."""
    return f"{arch}_seed{seed}_snap{k}of{n_snapshots}.pt"


def train_snapshot(args: argparse.Namespace) -> None:
    pool, val_indices = load_split(args.split_file)
    arch = f"resnet{args.depth}"
    n = args.n_snapshots
    total_epochs = n * args.cycle_epochs
    seeds = make_seeds(args.seed)

    print(f"[split] {args.split_file}  backbone pool={len(pool)}  val={len(val_indices)}")
    print(f"[snapshot] {arch}  seed={args.seed}  {n} snapshots × {args.cycle_epochs} epochs "
          f"= {total_epochs} epochs  init_seed={seeds['init']}  loader_seed={seeds['loader']}")

    seed_everything(seeds["init"])
    model = ResNet(depth=args.depth).to(args.device)
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

    accs: list[float] = []
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
            accs.append(acc)
            print(f"  [snapshot {k}/{n}] val_acc {acc:.3f}")
            _save(
                model,
                args.output_dir / snapshot_filename(arch, args.seed, k, n),
                {
                    "method": "snapshot",
                    "arch": arch,
                    "depth": args.depth,
                    "seed": args.seed,
                    "split_file": str(args.split_file),
                    "snapshot": k,
                    "n_snapshots": n,
                    "cycle_epochs": args.cycle_epochs,
                    "epoch": epoch + 1,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "val_acc": acc,
                },
            )

    summary = "  ".join(f"snap{i+1} {a:.3f}" for i, a in enumerate(accs))
    print(f"[done] seed={args.seed}  val_acc per snapshot:  {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one snapshot-ensemble trajectory of CIFAR ResNets"
    )
    parser.add_argument("--seed", type=int, required=True,
                        help="Trajectory seed; derives the init and loader seeds")
    parser.add_argument("--n-snapshots", type=int, default=5,
                        help="M: snapshots taken from this trajectory (default: %(default)s)")
    parser.add_argument("--cycle-epochs", type=int, default=40,
                        help="Epochs per LR cycle (total = n_snapshots × cycle_epochs)")
    parser.add_argument("--depth", type=int, default=110,
                        help="CIFAR ResNet depth, must be 6n+2 (default: %(default)s)")
    parser.add_argument("--split-file", type=Path, default=Path("splits/three_way_seed42.pt"),
                        help="3-way split file made by splits/generate.py")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Peak LR each cycle restarts to (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("backbone/checkpoints/pool"))
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    train_snapshot(args)


if __name__ == "__main__":
    main()
