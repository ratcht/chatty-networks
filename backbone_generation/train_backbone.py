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
from backbone_generation.stratified_subsets import make_controlled_overlap_stratified_subsets
from backbone_generation.resnet20 import ResNet20

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


def make_train_val_indices(
    data_root: str,
    val_per_class: int = 100,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Stratified split of the CIFAR-100 train set into 40k train / 10k val.

    The split is fixed by `seed` and shared across all conditions and backbones
    so every model validates against the same held-out examples.
    """
    base_set = datasets.CIFAR100(data_root, train=True, download=True)
    targets = np.array([t for _, t in base_set])
    classes = np.unique(targets)
    rng = np.random.default_rng(seed)

    train_indices, val_indices = [], []
    for cls in classes:
        cls_idx = np.flatnonzero(targets == cls).copy()
        rng.shuffle(cls_idx)
        val_indices.extend(cls_idx[:val_per_class].tolist())
        train_indices.extend(cls_idx[val_per_class:].tolist())

    return train_indices, val_indices


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


def train_backbone(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: str,
) -> nn.Module:
    model = model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc = _evaluate(model, val_loader, device)
            avg_loss = running_loss / len(train_loader)
            print(f"  epoch {epoch+1:>3}/{epochs}  loss {avg_loss:.4f}  val_acc {acc:.3f}")

    return model


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


def _save_split(
    train_indices: list[int],
    val_indices: list[int],
    split_seed: int,
    output_dir: Path,
) -> None:
    path = output_dir / f"validation_split_seed{split_seed}.pt"
    if path.exists():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"train_indices": train_indices, "val_indices": val_indices, "split_seed": split_seed}, path)
    print(f"  saved → {path}")


def make_d0_pair(
    seed1: int,
    seed2: int,
    split_seed: int,
    epochs: int,
    device: str,
    data_root: str,
    batch_size: int,
    output_dir: Path,
) -> None:
    train_indices, val_indices = make_train_val_indices(data_root, seed=split_seed)
    _save_split(train_indices, val_indices, split_seed, output_dir)
    print(f"[split] train={len(train_indices)}  val={len(val_indices)}")

    for seed in (seed1, seed2):
        seeds = make_seeds(seed)
        print(f"\n[D0] backbone seed={seed}  init_seed={seeds['init']}  loader_seed={seeds['loader']}")
        seed_everything(seeds["init"])
        model = ResNet20()
        train_loader, val_loader = make_cifar100_loaders(
            train_indices, val_indices, batch_size, seeds["loader"], data_root
        )
        train_backbone(model, train_loader, val_loader, epochs, device)
        _save(
            model,
            output_dir / f"d0_backbone_seed{seed}_epoch{epochs}.pt",
            {"condition": "d0", "seed": seed, "split_seed": split_seed, "epochs": epochs},
        )


def make_d1_pair(
    seed1: int,
    seed2: int,
    overlap_frac: float,
    split_seed: int,
    epochs: int,
    device: str,
    data_root: str,
    batch_size: int,
    output_dir: Path,
) -> None:
    train_indices, val_indices = make_train_val_indices(data_root, seed=split_seed)
    _save_split(train_indices, val_indices, split_seed, output_dir)
    print(f"[split] train pool={len(train_indices)}  val={len(val_indices)}")

    # Stratified subsets drawn from the 40k pool, not the full 50k
    all_targets = np.array(datasets.CIFAR100(data_root, train=True, download=True).targets)
    pool_targets = all_targets[train_indices]
    pool_sub1, pool_sub2 = make_controlled_overlap_stratified_subsets(
        pool_targets, frac=0.75, overlap_frac=overlap_frac, seed=seed1
    )
    # Remap pool-local indices back to absolute CIFAR-100 train indices
    indices1 = [train_indices[i] for i in pool_sub1]
    indices2 = [train_indices[i] for i in pool_sub2]

    actual_overlap = len(set(indices1) & set(indices2)) / len(train_indices)
    print(f"[D1] overlap={actual_overlap:.3f} (requested {overlap_frac}), "
          f"|subset1|={len(indices1)}  |subset2|={len(indices2)}")

    for seed, indices in ((seed1, indices1), (seed2, indices2)):
        seeds = make_seeds(seed)
        print(f"\n[D1] backbone seed={seed}  init_seed={seeds['init']}  loader_seed={seeds['loader']}")
        seed_everything(seeds["init"])
        model = ResNet20()
        train_loader, val_loader = make_cifar100_loaders(
            indices, val_indices, batch_size, seeds["loader"], data_root
        )
        train_backbone(model, train_loader, val_loader, epochs, device)
        _save(
            model,
            output_dir / f"d1_backbone_seed{seed}_epoch{epochs}.pt",
            {
                "condition": "d1", "seed": seed, "split_seed": split_seed,
                "overlap_frac": overlap_frac, "epochs": epochs,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet20 backbone pairs for CIFAR-100 ensembles")
    parser.add_argument("--condition", choices=["d0", "d1"], required=True)
    parser.add_argument("--seed1", type=int, required=True)
    parser.add_argument("--seed2", type=int, required=True)
    parser.add_argument("--overlap-frac", type=float, default=0.5,
                        help="D1 only: fraction of each class shared by both subsets (default: 0.5)")
    parser.add_argument("--split-seed", type=int, default=0,
                        help="Seed for the 40k/10k train/val split (default: 0)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("backbone_generation/checkpoints"))
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"condition={args.condition}  seed1={args.seed1}  seed2={args.seed2}  "
          f"epochs={args.epochs}  device={args.device}")

    if args.condition == "d0":
        make_d0_pair(
            args.seed1, args.seed2, args.split_seed, args.epochs, args.device,
            args.data_root, args.batch_size, args.output_dir,
        )
    else:
        make_d1_pair(
            args.seed1, args.seed2, args.overlap_frac, args.split_seed, args.epochs,
            args.device, args.data_root, args.batch_size, args.output_dir,
        )


if __name__ == "__main__":
    main()
