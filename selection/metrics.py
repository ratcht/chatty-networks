"""
The metric registry used to score candidate expert groups.

Every metric is a property of a *set* of experts evaluated on the validation
split, and is permutation-invariant — {a, b} scores the same as {b, a} — which
is why the search enumerates combinations rather than permutations.

Metrics are not computed independently. Each declares the `requires` set of
primitives it needs, where a primitive is a per-combination count over the
validation set, and the runner computes the union of primitives across every
requested metric exactly once. That is what makes `--metrics` subsetting cheap
and correct at the same time: asking for a derived metric like
`oracle_minus_bagging` pulls in its primitives without the caller naming them,
and asking only for the correctness-based metrics skips the GPU pass that the
probability-based ones need.

Each metric also declares:
  version   — bumping it invalidates that column alone, never the others
  direction — "max" or "min", so "the best group" is unambiguous

What the metrics mean. Each is the ceiling for a specific way of combining
experts, which is why they map onto the ensemble methods:

  bagging      fixed uniform weights — the untrained prob-averaging baseline
               every joint-trained method has to beat
  soft_oracle  per-example convex weights — a soft gate with a perfect gating
               function (not yet implemented, see below)
  oracle       per-example one-hot weights — hard routing with a perfect gate

Note that `oracle` is NOT an upper bound on ensemble accuracy: averaging can be
right when no single expert's argmax is, so `bagging > oracle` is possible and
is not a bug.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

# Per-combination counts over the validation set. Names are the keys of the
# dict handed to every metric's fn.
PRIMITIVES = ("n_oracle", "n_shared", "n_bagging", "n_soft_oracle")


@dataclass(frozen=True)
class Metric:
    name: str
    version: int
    direction: str
    requires: tuple[str, ...]
    fn: Callable[[dict[str, np.ndarray], int], np.ndarray]
    doc: str

    def compute(self, primitives: dict[str, np.ndarray], n_val: int) -> np.ndarray:
        return self.fn(primitives, n_val).astype(np.float32, copy=False)


def _rate(key: str) -> Callable[[dict[str, np.ndarray], int], np.ndarray]:
    return lambda prim, n_val: prim[key] / n_val


def _difference(a: str, b: str) -> Callable[[dict[str, np.ndarray], int], np.ndarray]:
    return lambda prim, n_val: (prim[a] - prim[b]) / n_val


_METRICS: tuple[Metric, ...] = (
    Metric(
        name="oracle",
        version=1,
        direction="max",
        requires=("n_oracle",),
        fn=_rate("n_oracle"),
        doc="Fraction of val examples at least one expert classifies correctly. "
            "The ceiling for hard routing with a perfect gate.",
    ),
    Metric(
        name="shared",
        version=1,
        direction="max",
        requires=("n_shared",),
        fn=_rate("n_shared"),
        doc="Fraction of val examples every expert classifies correctly. The "
            "redundant core the group agrees on; 'all agree and are right' "
            "reduces to this, since experts that are all correct necessarily "
            "agree.",
    ),
    Metric(
        name="bagging",
        version=1,
        direction="max",
        requires=("n_bagging",),
        fn=_rate("n_bagging"),
        doc="Accuracy of the uniform average of the experts' softmax outputs — "
            "the untrained prob-averaging baseline a joint-trained ensemble has "
            "to beat.",
    ),
    Metric(
        name="oracle_minus_shared",
        version=1,
        direction="max",
        requires=("n_oracle", "n_shared"),
        fn=_difference("n_oracle", "n_shared"),
        doc="Fraction of val examples some but not all experts get right — the "
            "region where the experts disagree in correctness and a combiner "
            "has actual work to do.",
    ),
    Metric(
        name="oracle_minus_bagging",
        version=1,
        direction="max",
        requires=("n_oracle", "n_bagging"),
        fn=_difference("n_oracle", "n_bagging"),
        doc="Exploitable headroom: how much a perfect hard router would add over "
            "naive averaging. The room a gate or a communication channel has to "
            "work in.",
    ),
    Metric(
        name="soft_oracle",
        version=1,
        direction="max",
        requires=("n_soft_oracle",),
        fn=_rate("n_soft_oracle"),
        doc="Fraction of val examples for which SOME convex combination of the "
            "experts' softmax outputs has the correct argmax — the ceiling for a "
            "soft gate with a perfect gating function. Orders of magnitude more "
            "expensive than the others; see the note below for the k limit.",
    ),
)

REGISTRY: dict[str, Metric] = {m.name: m for m in _METRICS}


# ---------------------------------------------------------------------------
# soft_oracle — why it is restricted to small k
# ---------------------------------------------------------------------------
#
# Soft oracle accuracy is the fraction of val examples for which SOME convex
# combination of the experts' softmax outputs has the correct argmax. Oracle
# accuracy is the same quantity restricted to one-hot weights, so
# soft_oracle >= oracle always, and >= bagging too (uniform weights are convex).
#
# It is also the one metric communication can legitimately exceed: messages
# change what the experts output rather than only how they are weighted, so a
# communicative ensemble scoring above its group's soft oracle is doing
# something no gate could.
#
# For one example with true label y it is a feasibility problem: does there
# exist w in the simplex with
#
#     sum_i w_i * (p_i[y] - p_i[c]) > 0   for every wrong class c
#
# equivalently, the value of the zero-sum game with payoff
# M[i, c] = p_i[y] - p_i[c] is positive. As an LP: maximise t subject to
# M^T w >= t, w in the simplex; feasible iff t* > 0.
#
# That is one LP per (example, combination), so it only becomes affordable
# after a ladder of cheap decisions:
#   1. oracle correct  -> true  (one-hot witness)
#   2. bagging correct -> true  (uniform witness)
#   3. some class beats y for EVERY expert -> false (no convex combination can
#      recover a class that wins unanimously)
#   4. otherwise an LP, over only the "threatening" classes — those with
#      p_i[c] >= p_i[y] for at least one expert. A class no expert prefers to y
#      is satisfied by every w, so dropping it cannot change the sign of the
#      game value.
#
# Measured on this pool (50 experts, 5000 val examples) at k=5, the ladder
# still leaves ~218 examples per combination needing a real LP, and 88% of
# those have five or more threatening classes, so the small-case geometric
# shortcuts almost never fire. At scipy's measured 1.4 ms per solve that is
# 4.66e8 solves, roughly 170 hours — hence _SOFT_ORACLE_MAX_K in select.py.
#
# What the same measurement showed about the metric itself: soft_oracle sits a
# near-constant 0.0044 above oracle (std 0.0010 over 60 random k=5 groups), so
# it ranks groups almost identically — Spearman 0.980 against oracle, 0.964
# across the deciles, with the best and worst groups unchanged. It earns its
# cost as a *ceiling* on the groups actually trained, not as a selection
# criterion; oracle is a near-perfect stand-in for the ranking.


def get(name: str) -> Metric:
    if name not in REGISTRY:
        raise SystemExit(
            f"unknown metric {name!r}. Available: {', '.join(sorted(REGISTRY))}"
        )
    return REGISTRY[name]


def resolve(names: list[str]) -> tuple[list[Metric], set[str]]:
    """Map requested metric names to their metrics and the primitives they need.

    The primitive set is the union over all requested metrics, so each one is
    computed once no matter how many metrics derive from it.
    """
    metrics = [get(n) for n in names]
    primitives: set[str] = set()
    for m in metrics:
        primitives.update(m.requires)
    unknown = primitives - set(PRIMITIVES)
    if unknown:
        raise SystemExit(f"metric declares unimplemented primitive(s): {sorted(unknown)}")
    return metrics, primitives


def all_names() -> list[str]:
    return sorted(REGISTRY)
