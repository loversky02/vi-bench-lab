"""Module #1 (cont.) — local scoring + silver key.

The official answers live on the V-Bench server, so you cannot self-score the real
key locally. Two things this module gives you:

1. :func:`score` — score a submission against *any* answer key (a real key if you
   ever obtain one, or the silver key below), aggregated per category / group / total
   to mirror the leaderboard columns.
2. :func:`build_silver_key` — a majority-vote "silver" key from an ensemble of strong
   models (via 9router). Lets you iterate a small model locally without spamming the
   real grader. HONEST CAVEAT: a silver key measures agreement-with-the-ensemble, not
   ground truth; treat it as a proxy, and confirm gains on the real leaderboard.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .dataset import Row, Track, DOMAIN_TO_CATEGORY, DOMAIN_TO_GROUP, classify_track
from .models import Model


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
@dataclass
class ScoreReport:
    total_scored: int = 0
    total_correct: int = 0
    by_domain: dict[str, dict] = field(default_factory=dict)
    by_category: dict[str, dict] = field(default_factory=dict)
    by_group: dict[str, dict] = field(default_factory=dict)

    @property
    def accuracy_micro(self) -> float:
        return self.total_correct / self.total_scored if self.total_scored else 0.0

    @property
    def accuracy_macro_category(self) -> float:
        accs = [v["acc"] for v in self.by_category.values() if v["n"]]
        return sum(accs) / len(accs) if accs else 0.0

    def as_dict(self) -> dict:
        return {
            "total_scored": self.total_scored,
            "total_correct": self.total_correct,
            "accuracy_micro": round(self.accuracy_micro, 4),
            "accuracy_macro_category": round(self.accuracy_macro_category, 4),
            "by_category": self.by_category,
            "by_group": self.by_group,
            "by_domain": self.by_domain,
        }


def _norm_mc(v: object) -> str:
    return str(v).strip().upper()[:1]


def _answer_correct(row: Row, predicted: object, gold: object) -> bool:
    track = classify_track(row)
    if track == Track.MC:
        return _norm_mc(predicted) == _norm_mc(gold)
    if track == Track.AGENTIC:
        # Opaque official rule; we score exact tool-name match as a transparent proxy.
        pn = predicted.get("name") if isinstance(predicted, dict) else None
        gn = gold.get("name") if isinstance(gold, dict) else gold
        return pn is not None and pn == gn
    return False


def _bump(bucket: dict, key: str, correct: bool) -> None:
    slot = bucket.setdefault(key, {"correct": 0, "n": 0, "acc": 0.0})
    slot["n"] += 1
    if correct:
        slot["correct"] += 1
    slot["acc"] = round(slot["correct"] / slot["n"], 4)


def score(
    submission: Iterable[dict],
    answer_key: dict,
    rows: Iterable[Row],
) -> ScoreReport:
    """Score ``submission`` (records of ``{"id", "answer"}``) against ``answer_key``.

    ``answer_key`` maps ``id -> gold`` (a letter for MC, or ``{"name": ...}`` for
    agentic). Ids absent from the key are ignored (unscored).
    """
    index = {r.id: r for r in rows}
    key = {int(k): v for k, v in answer_key.items()}
    report = ScoreReport()
    for rec in submission:
        rid = int(rec["id"])
        if rid not in key or rid not in index:
            continue
        row = index[rid]
        correct = _answer_correct(row, rec.get("answer"), key[rid])
        report.total_scored += 1
        report.total_correct += int(correct)
        _bump(report.by_domain, row.domain, correct)
        _bump(report.by_category, DOMAIN_TO_CATEGORY.get(row.domain, row.domain), correct)
        _bump(report.by_group, DOMAIN_TO_GROUP.get(row.domain, "unknown"), correct)
    return report


# --------------------------------------------------------------------------- #
# Silver key
# --------------------------------------------------------------------------- #
def _majority(votes: list[str]) -> str:
    # Counter.most_common keeps first-seen order on ties -> deterministic.
    return Counter(votes).most_common(1)[0][0]


def build_silver_key(
    rows: Iterable[Row],
    models: list[Model],
    track: Track = Track.MC,
) -> dict[int, object]:
    """Majority-vote silver key from an ensemble. MC only by default."""
    rows = [r for r in rows if classify_track(r) == track]
    key: dict[int, object] = {}
    for row in rows:
        if track == Track.MC:
            votes = [m.predict_mc(row) for m in models]
            key[row.id] = _majority(votes)
        elif track == Track.AGENTIC:
            votes = [str(m.predict_agentic(row).get("name", "")) for m in models]
            key[row.id] = {"name": _majority(votes)}
    return key


def save_key(key: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rid, ans in key.items():
            f.write(json.dumps({"id": rid, "answer": ans}, ensure_ascii=False) + "\n")
    return path


def load_key(path: str | Path) -> dict[int, object]:
    key: dict[int, object] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                key[int(o["id"])] = o["answer"]
    return key
