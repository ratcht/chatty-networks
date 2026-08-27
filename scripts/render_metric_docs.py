"""
Render ensemble/metric_docs.py's registry to docs/metrics.md for the MkDocs
site. Run on demand — it's a docs artifact, not something regenerated per
training run.

uv run python scripts/render_metric_docs.py
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from ensemble.metric_docs import REGISTRY, SUMMARY_REGISTRY, CATEGORIES, MetricDoc

_OUT = _REPO_ROOT / "docs" / "metrics.md"


def _render_metric(d: MetricDoc, heading: str = "###") -> list[str]:
    lines = [
        f"{heading} `{d.name}`", "",
        f"**Summary.** {d.summary}", "",
        f"**Computation.** {d.computation}", "",
        f"**Intent.** {d.intent}", "",
    ]
    if not d.context:
        lines += ["Tracked with no context — one series.", ""]
        return lines
    for ck in d.context:
        lines += [f"**Context — `{ck.name}`.** This name covers several series, one per value:", ""]
        lines += [f"| `{ck.name}` value | meaning |", "|---|---|"]
        lines += [f"| `{value}` | {meaning} |" for value, meaning in ck.values.items()]
        lines += [""]
    return lines


def _render_flat_section(title: str, intro: str, docs: dict[str, MetricDoc]) -> str:
    lines = [f"## {title}", "", intro, ""]
    for name in sorted(docs):
        lines += _render_metric(docs[name])
    return "\n".join(lines)


def _render_grouped_section(title: str, intro: str, docs: dict[str, MetricDoc]) -> str:
    """Per-run metrics grouped into conceptual blocks (CATEGORIES) rather
    than flattened alphabetically — each block is *what question the metrics
    in it answer*, not just a label, so it opens with its own blurb."""
    lines = [f"## {title}", "", intro, ""]
    by_category: dict[str, list[MetricDoc]] = {key: [] for key, _, _ in CATEGORIES}
    for d in docs.values():
        by_category[d.category].append(d)
    for key, category_title, blurb in CATEGORIES:
        lines += [f"### {category_title}", "", blurb, ""]
        for d in sorted(by_category[key], key=lambda d: d.name):
            lines += _render_metric(d, heading="####")
    return "\n".join(lines)


def render() -> str:
    return "\n".join([
        "# Aim metrics",
        "",
        "Every metric `ensemble/train.py` tracks to Aim, generated from "
        "`ensemble/metric_docs.py` — the same registry every run writes "
        "into `run[\"metric_docs\"]`, so this page and the dashboard never "
        "drift apart.",
        "",
        _render_grouped_section(
            "Per-run metrics",
            "Tracked within a single `communicative` training run, grouped "
            "by what question each answers.",
            REGISTRY,
        ),
        _render_flat_section(
            "Replicate-group summary metrics",
            "Tracked only on the distinguished `summary` run "
            "`ensemble/train.py summarize` (or `scripts/replicate_joint.py`) "
            "produces after a group of N_2 seed replicates all finish.",
            SUMMARY_REGISTRY,
        ),
    ])


def main() -> None:
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(render())
    print(f"wrote → {_OUT}")


if __name__ == "__main__":
    main()
