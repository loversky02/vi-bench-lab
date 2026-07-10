"""Module #1 — the eval harness.

Runs a model over the *scored* tracks (MC + agentic) and emits ``submission.jsonl``
in V-Bench format: one ``{"id": ..., "answer": ...}`` per line, matched by ``id``.

Answer encoding
---------------
* MC       — ``answer`` is a single choice letter, e.g. ``"B"``.
* AGENTIC  — ``answer`` is the chosen function call. The public site documents the
  track as "Type 2 - Agentic (function call)" but does not publish the exact JSON
  shape, so we default to ``{"name": ..., "arguments": {...}}`` and expose
  ``agentic_format="openai"`` (arguments as a JSON *string*) as an alternative.
  See docs/NEXT.md — confirm against the grader before an official submission.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from .dataset import Row, Track, SCORED_TRACKS, classify_track
from .models import Model


@dataclass
class RunStats:
    model: str
    total: int = 0
    by_track: dict[str, int] = field(default_factory=dict)
    by_domain: dict[str, int] = field(default_factory=dict)
    skipped: int = 0

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "total": self.total,
            "by_track": self.by_track,
            "by_domain": self.by_domain,
            "skipped": self.skipped,
        }


def _encode_agentic(call: dict, agentic_format: str) -> object:
    name = call.get("name", "")
    args = call.get("arguments", {})
    if agentic_format == "openai":
        return {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}
    return {"name": name, "arguments": args}


def predict_row(model: Model, row: Row, agentic_format: str = "object") -> object:
    """Return the ``answer`` value for a single row (dispatch by track)."""
    track = classify_track(row)
    if track == Track.MC:
        return model.predict_mc(row)
    if track == Track.AGENTIC:
        return _encode_agentic(model.predict_agentic(row), agentic_format)
    raise ValueError(f"row {row.id} is on the {track.value} track (not scored by the harness)")


def build_submission(
    rows: Iterable[Row],
    model: Model,
    tracks: Iterable[Track] = SCORED_TRACKS,
    limit: Optional[int] = None,
    agentic_format: str = "object",
    progress: Optional[Callable[[int, int], None]] = None,
) -> tuple[list[dict], RunStats]:
    """Run ``model`` over rows whose track is in ``tracks``; return (submissions, stats)."""
    tracks = set(tracks)
    rows = [r for r in rows if classify_track(r) in tracks]
    if limit is not None:
        rows = rows[:limit]

    stats = RunStats(model=getattr(model, "name", "model"))
    subs: list[dict] = []
    n = len(rows)
    for i, row in enumerate(rows, 1):
        try:
            answer = predict_row(model, row, agentic_format=agentic_format)
        except Exception:
            stats.skipped += 1
            continue
        subs.append({"id": row.id, "answer": answer})
        stats.total += 1
        stats.by_track[classify_track(row).value] = stats.by_track.get(classify_track(row).value, 0) + 1
        stats.by_domain[row.domain] = stats.by_domain.get(row.domain, 0) + 1
        if progress and (i % 200 == 0 or i == n):
            progress(i, n)
    return subs, stats


def write_submission(subs: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in subs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path
