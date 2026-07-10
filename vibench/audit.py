"""Module #3 — vi-bench-audit.

Honest-finding lane. Two inputs, no model calls required:

* ``leaderboard/snapshot.json`` — the 10-model x 16-category table scraped from
  vbench.ai. We derive saturation, discriminativeness, category↔total correlation,
  redundant category pairs, and per-model spikiness (e.g. V-LLM's sovereignty spike
  vs its reasoning dip).
* the public test set (optional) — domain distribution, choice-length mix, question
  length, agentic decoy count.

Everything here is $0 and deterministic.
"""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Iterable, Optional

from .dataset import Row, DOMAIN_TO_CATEGORY

DEFAULT_LEADERBOARD_PATH = Path(__file__).resolve().parent.parent / "leaderboard" / "snapshot.json"

# Category subsets used for the headline "narrow-domain spike" finding.
REASONING = ["Toán", "Lý", "Logic", "Văn"]
SOVEREIGNTY = ["Chính trị cơ bản", "Chính trị chuyên sâu", "Hate speech"]


# --------------------------------------------------------------------------- #
# Leaderboard parsing
# --------------------------------------------------------------------------- #
def load_leaderboard(path: str | Path = DEFAULT_LEADERBOARD_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        lb = json.load(f)
    cats = lb["categories"]
    cat_start = lb["columns"].index(cats[0])
    models = []
    for row in lb["rows"]:
        scores = {c: float(row[cat_start + i]) for i, c in enumerate(cats)}
        models.append(
            {
                "rank": row[0],
                "name": row[1],
                "org": row[2],
                "size": row[3],
                "total": float(row[4]),
                "scores": scores,
            }
        )
    lb["models"] = models
    return lb


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sqrt(sum((x - mx) ** 2 for x in xs))
    dy = sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else 0.0


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #
def category_stats(lb: dict) -> dict[str, dict]:
    out = {}
    for c in lb["categories"]:
        vals = [m["scores"][c] for m in lb["models"]]
        out[c] = {
            "mean": round(_mean(vals), 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
            "std": round(_std(vals), 2),
            "range": round(max(vals) - min(vals), 2),
        }
    return out


def saturation(lb: dict, hi: float = 85.0, lo: float = 45.0) -> dict:
    stats = category_stats(lb)
    return {
        "saturated": sorted([c for c, s in stats.items() if s["mean"] >= hi],
                            key=lambda c: -stats[c]["mean"]),
        "hard": sorted([c for c, s in stats.items() if s["mean"] <= lo],
                       key=lambda c: stats[c]["mean"]),
    }


def corr_with_total(lb: dict) -> dict[str, float]:
    totals = [m["total"] for m in lb["models"]]
    return {
        c: round(_pearson([m["scores"][c] for m in lb["models"]], totals), 3)
        for c in lb["categories"]
    }


def redundant_pairs(lb: dict, thr: float = 0.9) -> list[dict]:
    cats = lb["categories"]
    series = {c: [m["scores"][c] for m in lb["models"]] for c in cats}
    pairs = []
    for i, a in enumerate(cats):
        for b in cats[i + 1:]:
            r = _pearson(series[a], series[b])
            if r >= thr:
                pairs.append({"a": a, "b": b, "corr": round(r, 3)})
    return sorted(pairs, key=lambda p: -p["corr"])


def model_gaps(lb: dict) -> list[dict]:
    out = []
    for m in lb["models"]:
        s = m["scores"]
        cats = sorted(s, key=lambda c: s[c])
        out.append(
            {
                "name": m["name"],
                "org": m["org"],
                "total": m["total"],
                "spread_std": round(_std(list(s.values())), 2),
                "best": {"cat": cats[-1], "score": s[cats[-1]]},
                "worst": {"cat": cats[0], "score": s[cats[0]]},
                "sovereignty_mean": round(_mean([s[c] for c in SOVEREIGNTY]), 2),
                "reasoning_mean": round(_mean([s[c] for c in REASONING]), 2),
                "sovereignty_minus_reasoning": round(
                    _mean([s[c] for c in SOVEREIGNTY]) - _mean([s[c] for c in REASONING]), 2
                ),
            }
        )
    return sorted(out, key=lambda x: -x["sovereignty_minus_reasoning"])


def public_set_stats(rows: Iterable[Row]) -> dict:
    rows = list(rows)
    by_domain: dict[str, int] = {}
    choice_lens: dict[int, int] = {}
    agentic_decoys: list[int] = []
    qlens: list[int] = []
    for r in rows:
        by_domain[r.domain] = by_domain.get(r.domain, 0) + 1
        qlens.append(len(r.question))
        if r.choices:
            choice_lens[len(r.choices)] = choice_lens.get(len(r.choices), 0) + 1
        if r.function:
            agentic_decoys.append(len(r.function))
    qlens.sort()
    p = lambda q: qlens[min(len(qlens) - 1, int(q * len(qlens)))] if qlens else 0
    return {
        "total": len(rows),
        "by_domain": dict(sorted(by_domain.items(), key=lambda kv: -kv[1])),
        "choice_length_mix": dict(sorted(choice_lens.items())),
        "agentic_avg_candidates": round(_mean(agentic_decoys), 2) if agentic_decoys else 0,
        "agentic_max_candidates": max(agentic_decoys) if agentic_decoys else 0,
        "question_len": {"p50": p(0.5), "p90": p(0.9), "p99": p(0.99), "max": qlens[-1] if qlens else 0},
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def audit_report(
    lb: Optional[dict] = None,
    rows: Optional[Iterable[Row]] = None,
) -> dict:
    lb = lb or load_leaderboard()
    stats = category_stats(lb)
    sat = saturation(lb)
    corr = corr_with_total(lb)
    redun = redundant_pairs(lb)
    gaps = model_gaps(lb)

    most_disc = max(stats, key=lambda c: stats[c]["std"])
    least_disc = min(stats, key=lambda c: stats[c]["std"])
    top_gap = gaps[0]

    findings = [
        f"Saturated (mean≥85): {', '.join(sat['saturated']) or 'none'} "
        f"— little signal left to separate models.",
        f"Hardest frontier (mean≤45): {', '.join(sat['hard']) or 'none'} "
        f"— where the field is genuinely weak.",
        f"Most discriminative category: {most_disc} (std={stats[most_disc]['std']}); "
        f"least: {least_disc} (std={stats[least_disc]['std']}).",
        f"Narrow-domain spike: {top_gap['name']} scores sovereignty {top_gap['sovereignty_mean']} "
        f"vs reasoning {top_gap['reasoning_mean']} (gap +{top_gap['sovereignty_minus_reasoning']}).",
        f"Category most correlated with Tổng: "
        f"{max(corr, key=corr.get)} (r={max(corr.values())}); "
        f"least: {min(corr, key=corr.get)} (r={min(corr.values())}).",
    ]
    if redun:
        findings.append(
            f"{len(redun)} category pair(s) with r≥0.9 (candidate redundancy), "
            f"top: {redun[0]['a']}↔{redun[0]['b']} (r={redun[0]['corr']})."
        )

    report = {
        "n_models": len(lb["models"]),
        "n_categories": len(lb["categories"]),
        "category_stats": stats,
        "saturation": sat,
        "corr_with_total": corr,
        "redundant_pairs": redun,
        "model_gaps": gaps,
        "findings": findings,
    }
    if rows is not None:
        report["public_set"] = public_set_stats(rows)
    return report
