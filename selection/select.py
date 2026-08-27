"""
Stage 3: score every k-subset of the expert pool and pick a group.

The pool from stage 2 holds N_1 * M candidate experts. This exhaustively scores
all C(N, k) subsets of it on the validation split, caches one column per metric,
and emits a manifest that stage 4 consumes as its list of backbones. The test
set is never touched.

Scores are computed once and kept, rather than a "best group" being computed and
discarded, so selection stays a lever rather than a preprocessing step: with
every metric recorded for every combination you can pick groups spanning the
oracle-minus-bagging headroom range and ask whether a method's benefit scales
with it, instead of committing to a single group up front.

Examples
--------
uv run python selection/select.py compute --k 5 --metrics oracle bagging oracle_minus_bagging
uv run python selection/select.py rank --k 5 --metric oracle_minus_bagging
uv run python selection/select.py manifest --k 5 --metric oracle_minus_bagging --rank 0 \\
    --out selection/manifests/high_gap.json
uv run python selection/select.py manifest --k 5 --metric oracle_minus_bagging --percentile 100 \\
    --out selection/manifests/low_gap.json
"""

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linprog

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from selection import cache as C
from selection import metrics as M

_DEFAULT_CACHE = Path("selection/cache")
_DEFAULT_POOL = Path("backbone/checkpoints/pool")
_DEFAULT_SPLIT = Path("splits/three_way_seed42.pt")

# soft_oracle needs one LP per (combination, example) that survives the
# short-circuit ladder. Measured on this pool that is ~218 examples per
# combination at 1.4 ms each, so C(50,5) would take on the order of 170 hours
# while ranking groups at Spearman 0.980 against oracle — see the note in
# metrics.py. C(50,3) is 19,600 combinations and finishes in about an hour.
_SOFT_ORACLE_MAX_K = 4


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------

def compute_primitives(
    needed: set[str],
    probs: np.ndarray,
    labels: np.ndarray,
    k: int,
    combo_batch: int,
    val_chunk: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Exhaustively evaluate every k-combination, returning per-combination counts.

    Correctness-based primitives run over bit-packed correctness rows, where a
    combination's oracle count is a popcount of an OR and its shared count a
    popcount of an AND — the whole k=5 sweep takes seconds.

    The bagging primitive needs the actual averaged probabilities, so it gathers
    and sums the members' rows on the GPU in chunks over the validation set.
    That dominates the runtime by orders of magnitude.
    """
    n_experts, n_val, _ = probs.shape
    total = math.comb(n_experts, k)
    out = {p: np.empty(total, dtype=np.int32) for p in needed}

    packed = None
    if {"n_oracle", "n_shared"} & needed:
        correct = probs.argmax(2) == labels[None, :]
        packed = np.packbits(correct, axis=1)

    chunks: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    if "n_bagging" in needed:
        gpu_probs = torch.from_numpy(probs).to(device)
        gpu_labels = torch.from_numpy(labels).to(device)
        # Flattened per-chunk views are materialised once here rather than
        # re-sliced inside the combination loop, where the copy would be paid
        # tens of thousands of times.
        for start in range(0, n_val, val_chunk):
            stop = min(start + val_chunk, n_val)
            flat = gpu_probs[:, start:stop, :].reshape(n_experts, -1).contiguous()
            chunks.append((flat, gpu_labels[start:stop], stop - start))
        del gpu_probs

    combos = itertools.combinations(range(n_experts), k)
    pos = 0
    t0 = time.time()
    n_batches = math.ceil(total / combo_batch)
    for b in range(n_batches):
        batch = np.array(list(itertools.islice(combos, combo_batch)), dtype=np.int64)
        size = len(batch)

        if packed is not None:
            gathered = packed[batch]
            if "n_oracle" in needed:
                merged = np.bitwise_or.reduce(gathered, axis=1)
                out["n_oracle"][pos:pos + size] = np.bitwise_count(merged).sum(1, dtype=np.int32)
            if "n_shared" in needed:
                merged = np.bitwise_and.reduce(gathered, axis=1)
                out["n_shared"][pos:pos + size] = np.bitwise_count(merged).sum(1, dtype=np.int32)

        if "n_bagging" in needed:
            idx = torch.from_numpy(batch).to(device)
            n_correct = torch.zeros(size, dtype=torch.int32, device=device)
            for flat, chunk_labels, width in chunks:
                acc = flat[idx[:, 0]].clone()
                for j in range(1, k):
                    acc += flat[idx[:, j]]
                pred = acc.view(size, width, -1).argmax(-1)
                n_correct += (pred == chunk_labels).sum(1).to(torch.int32)
            out["n_bagging"][pos:pos + size] = n_correct.cpu().numpy()

        pos += size
        if (b + 1) % 100 == 0 or b + 1 == n_batches:
            done = time.time() - t0
            eta = done / (b + 1) * (n_batches - b - 1)
            print(f"  [{pos:>10}/{total}] {done:6.1f}s elapsed  eta {eta:5.1f}s", flush=True)

    return out


def _soft_feasible(margins: np.ndarray) -> bool:
    """Is there a convex combination of these experts whose argmax is correct?

    margins is (k, n_threatening) holding p_i[y] - p_i[c]. Feasible iff the
    zero-sum game with this payoff has a positive value, which is the LP
    maximise t subject to M^T w >= t, w in the simplex.
    """
    k, n_threat = margins.shape
    objective = np.zeros(k + 1)
    objective[-1] = -1.0                      # maximise t
    a_ub = np.hstack([-margins.T, np.ones((n_threat, 1))])
    a_eq = np.zeros((1, k + 1))
    a_eq[0, :k] = 1.0                         # weights sum to one
    result = linprog(
        objective, A_ub=a_ub, b_ub=np.zeros(n_threat), A_eq=a_eq, b_eq=[1.0],
        bounds=[(0.0, None)] * k + [(None, None)], method="highs",
    )
    return bool(result.x[-1] > 0)


def compute_soft_oracle(probs: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Per-combination count of examples some convex combination gets right.

    Runs serially rather than batched on the GPU because the inner step is an
    LP per surviving example, which is why this is capped at small k. Each
    combination first disposes of the easy examples: a member that is already
    correct is a one-hot witness, a correct uniform average is a uniform
    witness, and a class that beats the true label for every member is a proof
    that no convex combination can recover it.
    """
    n_experts, n_val, _ = probs.shape
    total = math.comb(n_experts, k)
    out = np.empty(total, dtype=np.int32)

    true_p = np.take_along_axis(probs, labels[None, :, None], axis=2)
    margin = true_p - probs                                  # (N, n_val, C)
    beats = margin <= 0
    beats[:, np.arange(n_val), labels] = False               # y never beats itself
    correct = ~beats.any(2)

    n_lp = 0
    t0 = time.time()
    for pos, combo in enumerate(itertools.combinations(range(n_experts), k)):
        members = list(combo)
        beats_sel, margin_sel = beats[members], margin[members]
        known = correct[members].any(0) | (probs[members].mean(0).argmax(1) == labels)
        unanimous = beats_sel.all(0).any(1)

        count = int(known.sum())
        for j in np.flatnonzero(~known & ~unanimous):
            threatening = np.flatnonzero(beats_sel[:, j, :].any(0))
            count += _soft_feasible(margin_sel[:, j, threatening])
            n_lp += 1
        out[pos] = count

        if (pos + 1) % 200 == 0 or pos + 1 == total:
            done = time.time() - t0
            eta = done / (pos + 1) * (total - pos - 1)
            print(f"  [{pos + 1:>10}/{total}] {done:7.1f}s elapsed  eta {eta:7.1f}s  "
                  f"{n_lp} LPs", flush=True)

    return out


def cmd_compute(args: argparse.Namespace) -> None:
    experts = C.discover_experts(args.pool_dir)
    val_indices, split_dig = C.read_split(args.split_file)
    pool_dig = C.pool_digest(experts)
    n_experts = len(experts)
    total = math.comb(n_experts, args.k)

    print(f"[select] pool={args.pool_dir}  {n_experts} experts  digest={pool_dig}")
    print(f"[select] split={args.split_file}  val={len(val_indices)}  digest={split_dig}")
    print(f"[select] k={args.k}  C({n_experts},{args.k})={total} combinations")

    cdir = C.columns_dir(args.cache_root, pool_dig, split_dig, args.k)
    meta = C.load_meta(cdir)

    requested = [M.get(name) for name in args.metrics]
    todo = []
    for metric in requested:
        recorded = meta["metrics"].get(metric.name)
        fresh = (
            recorded is not None
            and recorded.get("version") == metric.version
            and (cdir / f"{metric.name}.npy").exists()
        )
        if fresh and not args.force:
            print(f"[select] {metric.name}: cached at version {metric.version}, skipping")
        else:
            todo.append(metric)

    if not todo:
        print("[select] nothing to compute")
        return

    _, primitives = M.resolve([m.name for m in todo])
    if "n_soft_oracle" in primitives and args.k > _SOFT_ORACLE_MAX_K:
        raise SystemExit(
            f"soft_oracle is capped at k<={_SOFT_ORACLE_MAX_K}; you asked for k={args.k}.\n"
            f"  C({n_experts},{args.k}) combinations would need roughly "
            f"{int(total * 218 * 1.4e-3 / 3600)} hours of LP solving, and soft_oracle\n"
            f"  ranks groups at Spearman 0.980 against oracle, which is already cached.\n"
            f"  Compute it at k<={_SOFT_ORACLE_MAX_K}, or raise _SOFT_ORACLE_MAX_K if you mean it."
        )
    print(f"[select] computing {', '.join(m.name for m in todo)}  "
          f"(primitives: {', '.join(sorted(primitives))})")

    probs, labels = C.ensure_predictions(
        experts, val_indices, split_dig, args.cache_root, args.device, args.data_root
    )

    t0 = time.time()
    counts: dict[str, np.ndarray] = {}
    batched = primitives - {"n_soft_oracle"}
    if batched:
        counts.update(compute_primitives(
            batched, probs, labels, args.k,
            args.combo_batch, args.val_chunk, args.device,
        ))
    if "n_soft_oracle" in primitives:
        counts["n_soft_oracle"] = compute_soft_oracle(probs, labels, args.k)
    elapsed = time.time() - t0

    n_val = len(labels)
    for metric in todo:
        C.save_column(cdir, metric.name, metric.compute(counts, n_val))
        meta["metrics"][metric.name] = {"version": metric.version, "computed_at": C.now()}
    meta["n_combinations"] = total
    meta["n_val"] = n_val
    meta["k"] = args.k
    C.save_experts(cdir, experts)
    C.save_meta(cdir, meta)

    print(f"[select] wrote {len(todo)} column(s) to {cdir} in {elapsed:.1f}s")
    for metric in todo:
        col = C.load_column(cdir, metric.name)
        print(f"  {metric.name:<22} min {col.min():.4f}  mean {col.mean():.4f}  max {col.max():.4f}")


# ---------------------------------------------------------------------------
# rank
# ---------------------------------------------------------------------------

def _open_columns(args: argparse.Namespace) -> tuple[Path, list[C.Expert], dict]:
    experts = C.discover_experts(args.pool_dir)
    _, split_dig = C.read_split(args.split_file)
    cdir = C.columns_dir(args.cache_root, C.pool_digest(experts), split_dig, args.k)
    if not cdir.exists():
        raise SystemExit(
            f"no columns for this pool/split/k: {cdir}\n"
            f"  run: uv run python selection/select.py compute --k {args.k} --metrics ..."
        )
    return cdir, C.load_experts(cdir), C.load_meta(cdir)


def _report_rows(cdir: Path, experts: list[C.Expert], rows: np.ndarray,
                 primary: str, k: int, labels: list[str] | None = None) -> None:
    """Print selected combinations with every cached metric, not just the one
    sorted on — the cross-tabulation is the point of storing all of them."""
    available = C.cached_metrics(cdir)
    columns = {name: C.load_column(cdir, name) for name in available}
    ordered = [primary] + [n for n in available if n != primary]
    n_experts = len(experts)

    header = f"  {'#':>4}  {'row':>10}  " + "  ".join(f"{n:>20}" for n in ordered) + "   experts"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, row in enumerate(rows):
        combo = C.combination_at(int(row), n_experts, k)
        members = " ".join(experts[c].label for c in combo)
        values = "  ".join(f"{columns[n][row]:>20.4f}" for n in ordered)
        tag = labels[i] if labels else f"{i}"
        print(f"  {tag:>4}  {row:>10}  {values}   {members}")


def cmd_rank(args: argparse.Namespace) -> None:
    cdir, experts, meta = _open_columns(args)
    metric = M.get(args.metric)
    column = C.load_column(cdir, metric.name)
    total = len(column)

    print(f"[select] {cdir}")
    print(f"[select] {total} combinations  metric={metric.name} (direction={metric.direction})")
    print(f"[select] {metric.name}: min {column.min():.4f}  mean {column.mean():.4f}  "
          f"max {column.max():.4f}  std {column.std():.4f}")

    # The decile spread is the whole view: 0% is the best group and 100% the
    # worst, so it already carries both ends plus the shape in between. That is
    # what the headroom study wants — groups spanning the metric's range — and
    # the endpoints are the pair worth comparing, since the middle deciles sit
    # within a fraction of a point of each other.
    pcts = list(range(0, 101, 10))
    order = np.argsort(column, kind="stable")
    if metric.direction == "max":
        order = order[::-1]
    picks = [order[min(int(p / 100 * (total - 1)), total - 1)] for p in pcts]
    print(f"\n  decile spread (0% = best by direction={metric.direction})")
    _report_rows(cdir, experts, np.array(picks), metric.name, args.k,
                 labels=[f"{p}%" for p in pcts])


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def cmd_manifest(args: argparse.Namespace) -> None:
    cdir, experts, meta = _open_columns(args)
    metric = M.get(args.metric)
    column = C.load_column(cdir, metric.name)
    total = len(column)

    order = np.argsort(column, kind="stable")
    if metric.direction == "max":
        order = order[::-1]

    if args.percentile is not None:
        position = min(int(args.percentile / 100 * (total - 1)), total - 1)
        selector = {"percentile": args.percentile, "position": position}
    else:
        position = args.rank
        if position >= total:
            raise SystemExit(f"--rank {position} out of range (only {total} combinations)")
        selector = {"rank": position}

    row = int(order[position])
    combo = C.combination_at(row, len(experts), args.k)
    members = [experts[c] for c in combo]

    _, split_dig = C.read_split(args.split_file)
    manifest = {
        "created": C.now(),
        "k": args.k,
        "selected_by": metric.name,
        "direction": metric.direction,
        "metric_version": metric.version,
        **selector,
        "row": row,
        "combination": list(combo),
        "metrics": {
            name: float(C.load_column(cdir, name)[row]) for name in C.cached_metrics(cdir)
        },
        "pool_digest": C.pool_digest(experts),
        "split_digest": split_dig,
        "split_file": str(args.split_file),
        "pool_dir": str(args.pool_dir),
        "experts": [
            {"index": e.index, "path": e.path, "digest": e.digest,
             "seed": e.seed, "snapshot": e.snapshot, "val_acc": e.val_acc}
            for e in members
        ],
        # ready to paste into stage 4 as its list of backbones
        "backbones": [e.path for e in members],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[select] selected row {row} by {metric.name} ({list(selector)[0]}={list(selector.values())[0]})")
    for name, value in manifest["metrics"].items():
        print(f"  {name:<22} {value:.4f}")
    print(f"  experts: {' '.join(e.label for e in members)}")
    print(f"[select] wrote → {args.out}")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Exhaustive expert-group selection")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--k", type=int, required=True, help="Experts per group")
    common.add_argument("--pool-dir", type=Path, default=_DEFAULT_POOL)
    common.add_argument("--split-file", type=Path, default=_DEFAULT_SPLIT)
    common.add_argument("--cache-root", type=Path, default=_DEFAULT_CACHE)

    p_compute = sub.add_parser("compute", parents=[common],
                               help="Score every k-combination and cache the columns")
    p_compute.add_argument("--metrics", nargs="+", required=True,
                           metavar="NAME", help=f"Any of: {', '.join(M.all_names())}")
    p_compute.add_argument("--combo-batch", type=int, default=2048,
                           help="Combinations scored per batch (default: %(default)s)")
    p_compute.add_argument("--val-chunk", type=int, default=250,
                           help="Validation examples per inner chunk (default: %(default)s)")
    p_compute.add_argument("--data-root", type=str, default="data")
    p_compute.add_argument("--device", type=str, default="cuda")
    p_compute.add_argument("--force", action="store_true",
                           help="Recompute columns that are already cached")

    p_rank = sub.add_parser("rank", parents=[common],
                            help="Show the metric's decile spread over every group")
    p_rank.add_argument("--metric", type=str, required=True)

    p_manifest = sub.add_parser("manifest", parents=[common],
                                help="Write a selected group's manifest for stage 4")
    p_manifest.add_argument("--metric", type=str, required=True)
    p_manifest.add_argument("--rank", type=int, default=0,
                            help="0 is the best group by the metric's direction")
    p_manifest.add_argument("--percentile", type=float, default=None,
                            help="Select by percentile instead of rank (0 = best)")
    p_manifest.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")

    {"compute": cmd_compute, "rank": cmd_rank, "manifest": cmd_manifest}[args.command](args)


if __name__ == "__main__":
    main()
