"""
Generate the expert pool: one snapshot-ensemble trajectory per expert seed.

Stage 2 of the pipeline. Reads the expert seeds recorded in the split file by
splits/generate.py and runs backbone/train.py once per seed, producing
N_1 trajectories x M snapshots = N_1*M candidate experts for selection to
choose a group from.

This is a long job — a 200-epoch ResNet-110 trajectory is well over an hour,
and N_1 of them is most of a day — so it is resumable. Progress is tracked per
seed in a state file next to the checkpoints; rerunning the same command skips
trajectories that already finished and picks up at the first one that did not.
Resume granularity is the whole trajectory: a seed killed at snapshot 3 of 5
restarts from scratch, losing that trajectory's work but nothing else.

Examples
--------
uv run python scripts/generate_pool.py --dry-run
uv run python scripts/generate_pool.py
uv run python scripts/generate_pool.py --limit 2      # first 2 seeds only
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from backbone.train import snapshot_filename

_STATE_VERSION = 1

# Flags forwarded verbatim to backbone/train.py. Every one of them changes what
# a trajectory *is*, so they also form the config fingerprint below: resuming a
# pool with any of them altered would silently mix incomparable experts.
_TRAIN_FLAGS = ("depth", "n_snapshots", "cycle_epochs", "lr", "batch_size")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_split(split_file: Path) -> tuple[list[int], str]:
    """Return the expert seeds and a digest of the partition itself.

    The digest matters because splits/generate.py writes in place: bumping
    --n-expert-seeds rewrites the same path, and so does changing --fracs. The
    path alone can't tell those apart, so a repartitioned split would sail past
    the config guard and get half a pool trained on different data.
    """
    split = torch.load(split_file, map_location="cpu", weights_only=True)
    seeds = split.get("expert_seeds")
    if not seeds:
        raise SystemExit(
            f"{split_file} has no expert_seeds — it predates seed lists. "
            f"Regenerate it: uv run python splits/generate.py --seed {split.get('seed', 0)}"
        )
    h = hashlib.sha256()
    for key in ("backbone_indices", "ensemble_indices", "val_indices"):
        h.update(np.asarray(split[key], dtype=np.int64).tobytes())
    return [int(s) for s in seeds], h.hexdigest()[:16]


def config_fingerprint(args: argparse.Namespace, split_digest: str) -> dict:
    fp = {f: getattr(args, f) for f in _TRAIN_FLAGS}
    fp["split_file"] = str(args.split_file)
    fp["split_digest"] = split_digest
    return fp


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"version": _STATE_VERSION, "config": None, "seeds": {}}
    with path.open() as f:
        return json.load(f)


def save_state(path: Path, state: dict) -> None:
    """Atomic, for the same reason checkpoints are: a state file truncated by a
    kill would lose the record of every finished trajectory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def is_complete(seed: int, args: argparse.Namespace, state: dict) -> bool:
    """A trajectory counts as done only if the state file says so *and* every
    snapshot it should have produced is still on disk — so deleting a
    checkpoint is enough to queue its trajectory for a rerun."""
    entry = state["seeds"].get(str(seed))
    if not entry or entry.get("status") != "done":
        return False
    arch = f"resnet{args.depth}"
    return all(
        (args.output_dir / snapshot_filename(arch, seed, k, args.n_snapshots)).exists()
        for k in range(1, args.n_snapshots + 1)
    )


def build_command(seed: int, args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, str(_REPO_ROOT / "backbone" / "train.py"), "--seed", str(seed)]
    for flag in _TRAIN_FLAGS:
        cmd += [f"--{flag.replace('_', '-')}", str(getattr(args, flag))]
    cmd += [
        "--split-file", str(args.split_file),
        "--output-dir", str(args.output_dir),
        "--data-root", args.data_root,
        "--device", args.device,
    ]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the N_1 x M expert pool")
    parser.add_argument("--split-file", type=Path, default=Path("splits/three_way_seed42.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("backbone/checkpoints/pool"))
    parser.add_argument("--state-file", type=Path, default=None,
                        help="Default: <output-dir>/pool_state.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override the split file's expert seeds")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N seeds (the pool grows by appending)")
    parser.add_argument("--n-snapshots", type=int, default=5, help="M per trajectory")
    parser.add_argument("--cycle-epochs", type=int, default=40)
    parser.add_argument("--depth", type=int, default=110)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run, touch nothing")
    parser.add_argument("--stop-on-failure", action="store_true",
                        help="Abort on the first failed trajectory (default: carry on)")
    parser.add_argument("--force", action="store_true",
                        help="Proceed even if the state file records a different config")
    args = parser.parse_args()

    args.output_dir = (_REPO_ROOT / args.output_dir).resolve()
    args.split_file = (_REPO_ROOT / args.split_file).resolve()
    state_file = args.state_file or args.output_dir / "pool_state.json"

    expert_seeds, digest = read_split(args.split_file)
    seeds = args.seeds or expert_seeds
    if args.seeds:
        print("[pool] warning: --seeds overrides the split file's expert seeds, so this "
              "pool is no longer reproducible from the master --seed alone")
    if args.limit is not None:
        seeds = seeds[:args.limit]

    state = load_state(state_file)
    fingerprint = config_fingerprint(args, digest)
    if state["config"] and state["config"] != fingerprint and not args.force:
        diff = {k: (state["config"].get(k), v) for k, v in fingerprint.items()
                if state["config"].get(k) != v}
        raise SystemExit(
            f"config differs from the pool already in {state_file}\n"
            + "\n".join(f"  {k}: recorded {was!r}, now {now!r}" for k, (was, now) in diff.items())
            + "\nTrajectories trained under different settings are not comparable "
              "experts. Use a fresh --output-dir, or --force if you mean it."
        )

    todo = [s for s in seeds if not is_complete(s, args, state)]
    done = len(seeds) - len(todo)
    total_epochs = args.n_snapshots * args.cycle_epochs
    print(f"[pool] resnet{args.depth}  {len(seeds)} seeds × {args.n_snapshots} snapshots "
          f"× {args.cycle_epochs} epochs ({total_epochs} epochs/trajectory)")
    print(f"[pool] output={args.output_dir}")
    print(f"[pool] {done} already complete, {len(todo)} to run")

    if args.dry_run:
        for i, seed in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] seed={seed}\n  {' '.join(build_command(seed, args))}")
        return
    if not todo:
        print("[pool] nothing to do")
        return

    state["config"] = fingerprint
    results: dict[int, str] = {}
    for i, seed in enumerate(todo, 1):
        cmd = build_command(seed, args)
        print(f"\n{'=' * 70}\n[{i}/{len(todo)}] seed={seed}\n{'=' * 70}", flush=True)
        state["seeds"][str(seed)] = {"status": "running", "started": _now()}
        save_state(state_file, state)

        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=_REPO_ROOT)
            rc = proc.returncode
        except KeyboardInterrupt:
            state["seeds"][str(seed)] = {"status": "interrupted", "at": _now()}
            save_state(state_file, state)
            print(f"\n[pool] interrupted during seed={seed}; it will rerun from scratch. "
                  f"{i - 1} of {len(todo)} finished this session.")
            raise SystemExit(130)

        elapsed = time.time() - t0
        status = "done" if rc == 0 else "failed"
        results[seed] = status
        entry = {"status": status, "finished": _now(), "elapsed_sec": round(elapsed, 1)}
        if rc != 0:
            entry["returncode"] = rc
        state["seeds"][str(seed)] = entry
        save_state(state_file, state)
        print(f"[{i}/{len(todo)}] seed={seed} {status} in {elapsed / 60:.1f} min")

        if rc != 0 and args.stop_on_failure:
            print("[pool] --stop-on-failure set, aborting")
            break

    print(f"\n{'=' * 70}\n[pool] summary")
    for seed, status in results.items():
        print(f"  seed {seed:>12}  {status}")
    failed = [s for s, st in results.items() if st != "done"]
    skipped = len(todo) - len(results)
    if skipped:
        print(f"  ({skipped} not attempted)")
    if failed:
        raise SystemExit(f"[pool] {len(failed)} trajectory(ies) failed: {failed}")
    print("[pool] all trajectories complete")

if __name__ == "__main__":
    main()
