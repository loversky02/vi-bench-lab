"""Dataset loading, V-Bench taxonomy, and track routing.

The public test set (``data/public-test.jsonl``) has one JSON object per line with
fields: ``id``, ``question``, ``choices`` (list), ``function`` (list), ``domain``.
Answers are NOT included — scoring happens on the V-Bench server (or against a local
silver key; see ``scorer.py``).

A row belongs to exactly one *track*, inferred from its structure:
  * AGENTIC     — ``function`` is non-empty  (tool-selection + arg-grounding)
  * MC          — ``choices`` is non-empty   (multiple-choice knowledge)
  * GENERATION  — neither                     (open generation = Safety track)

V-Bench scores MC + AGENTIC ("Active"); the GENERATION rows are the Safety track it
currently leaves unscored — which is exactly what the ``safety`` module fills in.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "public-test.jsonl"


# --------------------------------------------------------------------------- #
# Taxonomy: public-test `domain` -> leaderboard category / data group.
# Mirrors the 16 fine-grained columns and 5 groups shown on vbench.ai.
# --------------------------------------------------------------------------- #
DOMAIN_TO_CATEGORY: dict[str, str] = {
    "mathematics": "Toán",
    "physics": "Lý",
    "chemistry": "Hóa",
    "computer_science": "CS",
    "literature": "Văn",
    "psychology": "Tâm lý",
    "philosophy": "Triết",
    "laws": "Luật",
    "logics": "Logic",
    "politics_easy": "Chính trị cơ bản",
    "politics_advanced": "Chính trị chuyên sâu",
    "hatespeech": "Hate speech",
    "dialect": "Phương ngữ",
    "culture": "Văn hóa VN",
    "medicine": "Sức khoẻ",
    "agentic": "Agentic",
}

_GROUP_DOMAINS: dict[str, tuple[str, ...]] = {
    "Kiến thức học thuật": (
        "mathematics", "physics", "chemistry", "computer_science",
        "literature", "psychology", "philosophy", "laws", "logics",
    ),
    "An toàn và chủ quyền số": ("politics_easy", "politics_advanced", "hatespeech"),
    "Tri thức văn hóa, xã hội, vùng miền": ("dialect", "culture"),
    "Y tế và sức khỏe cộng đồng": ("medicine",),
    "Agentic": ("agentic",),
}
DOMAIN_TO_GROUP: dict[str, str] = {
    d: g for g, domains in _GROUP_DOMAINS.items() for d in domains
}


class Track(str, Enum):
    MC = "mc"
    AGENTIC = "agentic"
    GENERATION = "generation"


#: Tracks V-Bench currently scores on the public leaderboard.
SCORED_TRACKS: frozenset[Track] = frozenset({Track.MC, Track.AGENTIC})


_LETTER_RE = re.compile(r"^\s*([A-Ea-e])\s*[\.\)\:\-]")


def choice_letters(choices: list[str]) -> list[str]:
    """Return the answer letter for each choice.

    Uses an explicit ``"A. ..."`` prefix when present, else assigns A, B, C… by
    position. Falls back gracefully so callers can always map an index to a letter.
    """
    letters: list[str] = []
    for i, c in enumerate(choices):
        m = _LETTER_RE.match(str(c))
        letters.append(m.group(1).upper() if m else chr(ord("A") + i))
    # de-duplicate defensively (e.g. malformed prefixes) by falling back to index
    if len(set(letters)) != len(letters):
        letters = [chr(ord("A") + i) for i in range(len(choices))]
    return letters


@dataclass(slots=True)
class Row:
    id: int
    question: str
    domain: str
    choices: list[str] = field(default_factory=list)
    function: list[dict] = field(default_factory=list)

    @classmethod
    def from_json(cls, o: dict) -> "Row":
        return cls(
            id=o["id"],
            question=o.get("question", ""),
            domain=o.get("domain", "unknown"),
            choices=list(o.get("choices") or []),
            function=list(o.get("function") or []),
        )

    @property
    def track(self) -> Track:
        return classify_track(self)

    @property
    def category(self) -> str:
        return DOMAIN_TO_CATEGORY.get(self.domain, self.domain)

    @property
    def group(self) -> str:
        return DOMAIN_TO_GROUP.get(self.domain, "unknown")

    @property
    def letters(self) -> list[str]:
        return choice_letters(self.choices)

    def format_choices(self) -> str:
        """Choices rendered for a prompt, one per line (kept verbatim)."""
        return "\n".join(str(c) for c in self.choices)


def classify_track(row: Row) -> Track:
    if row.function:
        return Track.AGENTIC
    if row.choices:
        return Track.MC
    return Track.GENERATION


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> list[Row]:
    """Load the public test set into a list of :class:`Row`."""
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(Row.from_json(json.loads(line)))
    return rows


def iter_dataset(path: str | Path = DEFAULT_DATA_PATH) -> Iterator[Row]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Row.from_json(json.loads(line))


def filter_track(rows: Iterable[Row], track: Track) -> list[Row]:
    return [r for r in rows if classify_track(r) == track]
