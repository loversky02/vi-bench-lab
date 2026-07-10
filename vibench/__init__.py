"""vi-bench-lab — an open, $0/Mac toolkit around the VinUni V-Bench (vbench.ai).

Three modules over one shared substrate (the public test set + leaderboard):
  1. harness  — run any model over the scored tracks (MC + agentic) -> submission.jsonl
  2. safety   — score the Safety track V-Bench does not auto-score (refusal + stance)
  3. audit    — meta-analysis of the leaderboard + public set (honest-finding lane)

Core is stdlib-only. Live model calls go through any OpenAI-compatible endpoint.
"""

__version__ = "0.1.0"

from .dataset import (  # noqa: F401
    Row,
    Track,
    load_dataset,
    classify_track,
    choice_letters,
    DOMAIN_TO_CATEGORY,
    DOMAIN_TO_GROUP,
    SCORED_TRACKS,
    DEFAULT_DATA_PATH,
)
