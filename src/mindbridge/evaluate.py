"""Offline evaluation: retrieval quality and crisis-detection accuracy.

Neither check needs an LLM or an API key. Everything is split into `dev` and `test`:
thresholds and weights are chosen on dev only, and test numbers are reported as they come.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .index import HybridRetriever, Mode
from .safety import Level, assess

MODES: tuple[Mode, ...] = ("bm25", "dense", "hybrid")
_LEVEL_NAMES = {"none": Level.NONE, "elevated": Level.ELEVATED, "high": Level.HIGH}


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def retrieval_metrics(retriever: HybridRetriever, rows: Iterable[dict], mode: Mode) -> dict:
    scored = [r for r in rows if r["relevant"]]
    hit = Counter()
    rr_total = 0.0
    for row in scored:
        relevant = set(row["relevant"])
        # one chunk per document so that ranks are document ranks
        docs = [h.chunk.doc_id for h in retriever.search(row["query"], k=10, mode=mode, per_doc=1)]
        rank = next((i + 1 for i, d in enumerate(docs) if d in relevant), None)
        if rank:
            rr_total += 1.0 / rank
            for k in (1, 3, 5):
                if rank <= k:
                    hit[k] += 1
    n = len(scored)
    return {
        "n": n,
        "hit@1": hit[1] / n,
        "hit@3": hit[3] / n,
        "hit@5": hit[5] / n,
        "mrr@10": rr_total / n,
    }


def scope_metrics(retriever: HybridRetriever, rows: list[dict], min_coverage: float) -> dict:
    in_scope = [r for r in rows if r["relevant"]]
    out_scope = [r for r in rows if not r["relevant"]]
    accepted = sum(retriever.coverage(r["query"]) >= min_coverage for r in in_scope)
    rejected = sum(retriever.coverage(r["query"]) < min_coverage for r in out_scope)
    return {
        "threshold": min_coverage,
        "in_scope_accepted": accepted / len(in_scope),
        "out_of_scope_rejected": rejected / len(out_scope),
        "n_in": len(in_scope),
        "n_out": len(out_scope),
    }


def calibrate_coverage(retriever: HybridRetriever, dev_rows: list[dict]) -> float:
    """Choose the coverage cutoff with the best balanced accuracy on dev.

    When several cutoffs tie, take the middle of the plateau rather than an edge.
    """
    ins = [retriever.coverage(r["query"]) for r in dev_rows if r["relevant"]]
    outs = [retriever.coverage(r["query"]) for r in dev_rows if not r["relevant"]]
    scored: list[tuple[float, float]] = []
    for step in range(10, 96):
        t = step / 100
        acc = (sum(s >= t for s in ins) / len(ins) + sum(s < t for s in outs) / len(outs)) / 2
        scored.append((acc, t))
    top = max(a for a, _ in scored)
    plateau = sorted(t for a, t in scored if abs(a - top) < 1e-9)
    return plateau[len(plateau) // 2]


def safety_metrics(rows: Iterable[dict]) -> dict:
    labels = ("none", "elevated", "high")
    confusion = {a: Counter() for a in labels}
    misses: list[tuple[str, str, str]] = []
    other_ok = other_total = 0
    for row in rows:
        result = assess(row["text"])
        predicted = result.level.name.lower()
        confusion[row["label"]][predicted] += 1
        if predicted != row["label"]:
            misses.append((row["label"], predicted, row["text"]))
        if row.get("about_other"):
            other_total += 1
            other_ok += int(result.about_other)

    def recall(label: str) -> float:
        total = sum(confusion[label].values())
        return confusion[label][label] / total if total else float("nan")

    def precision(label: str) -> float:
        predicted = sum(confusion[t][label] for t in labels)
        return confusion[label][label] / predicted if predicted else float("nan")

    flagged_none = sum(v for k, v in confusion["none"].items() if k != "none")
    n_none = sum(confusion["none"].values())
    # any-risk recall: the case counts as caught if it was flagged at either level
    risky = [r for r in labels if r != "none"]
    caught = sum(v for r in risky for k, v in confusion[r].items() if k != "none")
    n_risky = sum(sum(confusion[r].values()) for r in risky)
    return {
        "confusion": {a: dict(c) for a, c in confusion.items()},
        "recall_high": recall("high"),
        "precision_high": precision("high"),
        "recall_elevated": recall("elevated"),
        "any_risk_recall": caught / n_risky,
        "false_flag_rate": flagged_none / n_none,
        "about_other_correct": (other_ok, other_total),
        "misses": misses,
    }


def run_all(
    retriever: HybridRetriever, retrieval_rows: list[dict], safety_rows: list[dict]
) -> dict:
    dev = [r for r in retrieval_rows if r["split"] == "dev"]
    test = [r for r in retrieval_rows if r["split"] == "test"]
    threshold = calibrate_coverage(retriever, dev)
    return {
        "backend": retriever.embedder.name,
        "n_chunks": len(retriever.chunks),
        "retrieval": {
            split: {m: retrieval_metrics(retriever, rows, m) for m in MODES}
            for split, rows in (("dev", dev), ("test", test))
        },
        "scope": {
            "dev": scope_metrics(retriever, dev, threshold),
            "test": scope_metrics(retriever, test, threshold),
        },
        "safety": {
            split: safety_metrics([r for r in safety_rows if r["split"] == split])
            for split in ("dev", "test")
        },
    }


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{x * 100:.0f}%"


def render_markdown(res: dict) -> str:
    out = [
        "# Evaluation report",
        "",
        "Generated by `mindbridge eval`. Thresholds are calibrated on the dev split only; "
        "test numbers are reported as measured.",
        "",
        f"- Embedding backend: `{res['backend']}`",
        f"- Indexed chunks: {res['n_chunks']}",
        "",
        "## Retrieval (document level)",
        "",
        "Queries are hand-written in everyday phrasing and labelled with the document(s) that "
        "should answer them. `hit@k` is the share of queries with a relevant document in the "
        "top k.",
        "",
    ]
    for split in ("dev", "test"):
        n = res["retrieval"][split]["hybrid"]["n"]
        out += [
            f"### {split} ({n} queries)",
            "",
            "| retriever | hit@1 | hit@3 | hit@5 | MRR@10 |",
            "|---|---|---|---|---|",
        ]
        for mode in MODES:
            m = res["retrieval"][split][mode]
            out.append(
                f"| {mode} | {_pct(m['hit@1'])} | {_pct(m['hit@3'])} | {_pct(m['hit@5'])} | "
                f"{m['mrr@10']:.2f} |"
            )
        out.append("")

    out += ["## Out-of-scope handling", ""]
    out += [
        "| split | coverage cutoff | in-scope answered | out-of-scope declined |",
        "|---|---|---|---|",
    ]
    for split in ("dev", "test"):
        s = res["scope"][split]
        out.append(
            f"| {split} | {s['threshold']:.2f} | {_pct(s['in_scope_accepted'])} "
            f"({s['n_in']}) | {_pct(s['out_of_scope_rejected'])} ({s['n_out']}) |"
        )
    out += ["", "## Crisis detection", ""]
    out += [
        "| split | HIGH recall | HIGH precision | ELEVATED recall | any-risk recall | "
        "false flags on neutral text |",
        "|---|---|---|---|---|---|",
    ]
    for split in ("dev", "test"):
        s = res["safety"][split]
        out.append(
            f"| {split} | {_pct(s['recall_high'])} | {_pct(s['precision_high'])} | "
            f"{_pct(s['recall_elevated'])} | {_pct(s['any_risk_recall'])} | "
            f"{_pct(s['false_flag_rate'])} |"
        )
    out.append("")
    for split in ("dev", "test"):
        misses = res["safety"][split]["misses"]
        if misses:
            out += [f"Misclassified in {split}:", ""]
            out += [f'- expected `{e}`, got `{g}`: "{t}"' for e, g, t in misses]
            out.append("")
    out += [
        "## Caveats",
        "",
        "- The sets are small and written by the author of the system, so they flatter it. "
        "Treat the numbers as a regression guard, not a benchmark.",
        "- Generation quality (groundedness, tone) is not covered here because it needs a live "
        "model; see docs/design-decisions.md for how it is checked manually.",
        "",
    ]
    return "\n".join(out)
