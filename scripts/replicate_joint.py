"""
Replicate a joint-ensemble training config across N_2 seeds and aggregate the
result. Stage 4 of the pipeline: stages 1-3 rule out seed sensitivity in the
*backbone* pool (the split file's expert_seeds, scripts/generate_pool.py,
selection/select.py); this rules it out for the *joint-trained* ensemble
itself, using the same split file's joint_seeds.

This script is method-agnostic on purpose: it never inspects which method or
flags follow `--`, only runs that block once per seed. Pass the whole
`ensemble/train.py <method> ...` invocation after a literal `--`; everything
in it is forwarded verbatim except --seed (looped) and --experiment (fixed
for the whole group, so all N_2 runs land as multiple Runs in one Aim
experiment rather than N_2 separate experiments).

Long job, so resumable — coarse, per-seed granularity: a seed killed mid-run
restarts from scratch, others are untouched. Mirrors
scripts/generate_pool.py's state-file/config-fingerprint pattern.

Examples
--------
uv run python scripts/replicate_joint.py --experiment comm_group_a --dry-run -- \\
    communicative --manifest selection/manifests/high_gap.json --epochs 10

uv run python scripts/replicate_joint.py --experiment comm_group_a -- \\
    communicative --manifest selection/manifests/high_gap.json --epochs 10

uv run python scripts/replicate_joint.py --experiment comm_group_a --aggregate-only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from splits.generate import read_seeds

_STATE_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Everything before the first literal `--` is this script's own args;
    everything after is opaque and forwarded verbatim to ensemble/train.py."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def config_fingerprint(trailing: list[str], experiment: str, split_digest: str) -> dict:
    return {"trailing": trailing, "experiment": experiment, "split_digest": split_digest}


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"version": _STATE_VERSION, "config": None, "seeds": {}}
    with path.open() as f:
        return json.load(f)


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def is_complete(seed: int, state: dict) -> bool:
    return state["seeds"].get(str(seed), {}).get("status") == "done"


def build_command(seed: int, trailing: list[str], experiment: str) -> list[str]:
    # appended at the end so these win over anything accidentally duplicated
    # in the trailing block — argparse takes the last occurrence of a flag
    return [sys.executable, str(_REPO_ROOT / "ensemble" / "train.py"),
            *trailing, "--seed", str(seed), "--experiment", experiment]


def main() -> None:
    own_argv, trailing = split_argv(sys.argv[1:])

    parser = argparse.ArgumentParser(
        description="Replicate a joint-ensemble config across N_2 seeds and aggregate the result",
    )
    parser.add_argument("--split-file", type=Path, default=Path("splits/three_way_seed42.pt"))
    parser.add_argument("--experiment", required=True,
                        help="Aim experiment name shared by every replicate run")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override the split file's joint seeds")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N seeds")
    parser.add_argument("--state-file", type=Path, default=None,
                        help="Default: scripts/replicate_state/<experiment>.json")
    parser.add_argument("--repo", type=str, default=None,
                        help="Aim repo path for aggregation (default: Aim's default repo)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run, touch nothing")
    parser.add_argument("--stop-on-failure", action="store_true",
                        help="Abort on the first failed seed (default: carry on)")
    parser.add_argument("--force", action="store_true",
                        help="Proceed even if the state file records a different config")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip training; aggregate an already-done sweep's runs "
                             "(or re-aggregate after adding more seeds)")
    args = parser.parse_args(own_argv)

    if not trailing and not args.aggregate_only:
        parser.error("missing trailing `-- <method> <flags...>` block to forward to "
                      "ensemble/train.py (e.g. `-- communicative --manifest ... --epochs 10`)")

    args.split_file = (_REPO_ROOT / args.split_file).resolve()
    state_file = args.state_file or _REPO_ROOT / "scripts" / "replicate_state" / f"{args.experiment}.json"

    joint_seeds, digest = read_seeds(args.split_file, "joint_seeds")
    seeds = args.seeds or joint_seeds
    if args.seeds:
        print("[replicate] warning: --seeds overrides the split file's joint seeds, so this "
              "group is no longer reproducible from the master --seed alone")
    if args.limit is not None:
        seeds = seeds[:args.limit]

    if args.aggregate_only:
        from ensemble.train import summarize_replicates
        run = summarize_replicates(args.experiment, seeds, repo=args.repo)
        print(f"[replicate] wrote summary run {run.hash} in experiment {args.experiment!r} "
              f"over seeds {seeds}")
        return

    state = load_state(state_file)
    fingerprint = config_fingerprint(trailing, args.experiment, digest)
    if state["config"] and state["config"] != fingerprint and not args.force:
        diff = {k: (state["config"].get(k), v) for k, v in fingerprint.items()
                if state["config"].get(k) != v}
        raise SystemExit(
            f"config differs from the group already in {state_file}\n"
            + "\n".join(f"  {k}: recorded {was!r}, now {now!r}" for k, (was, now) in diff.items())
            + "\nReplicates trained under different settings are not comparable seeds "
              "of the same config. Use a fresh --experiment/--state-file, or --force if "
              "you mean it."
        )

    todo = [s for s in seeds if not is_complete(s, state)]
    done = len(seeds) - len(todo)
    print(f"[replicate] experiment={args.experiment!r}  {len(seeds)} seeds  "
          f"trailing: {' '.join(trailing)}")
    print(f"[replicate] {done} already complete, {len(todo)} to run")

    if args.dry_run:
        for i, seed in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] seed={seed}\n  {' '.join(build_command(seed, trailing, args.experiment))}")
        return
    if not todo:
        print("[replicate] nothing to do")
    else:
        state["config"] = fingerprint
        results: dict[int, str] = {}
        for i, seed in enumerate(todo, 1):
            cmd = build_command(seed, trailing, args.experiment)
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
                print(f"\n[replicate] interrupted during seed={seed}; it will rerun from "
                      f"scratch. {i - 1} of {len(todo)} finished this session.")
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
                print("[replicate] --stop-on-failure set, aborting")
                break

        print(f"\n{'=' * 70}\n[replicate] summary")
        for seed, status in results.items():
            print(f"  seed {seed:>12}  {status}")
        failed = [s for s, st in results.items() if st != "done"]
        skipped = len(todo) - len(results)
        if skipped:
            print(f"  ({skipped} not attempted)")
        if failed:
            raise SystemExit(f"[replicate] {len(failed)} seed(s) failed: {failed}")

    if any(not is_complete(s, state) for s in seeds):
        print("[replicate] not every seed is done — skipping aggregation. Rerun once "
              "they all are, or pass --aggregate-only later.")
        return

    print("[replicate] all seeds complete — aggregating")
    from ensemble.train import summarize_replicates
    run = summarize_replicates(args.experiment, seeds, repo=args.repo)
    print(f"[replicate] wrote summary run {run.hash} in experiment {args.experiment!r}")


if __name__ == "__main__":
    main()
