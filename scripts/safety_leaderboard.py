#!/usr/bin/env python3
"""Score the Safety track V-Bench marks Inactive, across several models via an
OpenAI-compatible gateway → the safety leaderboard the site does not ship.

Defensive by construction: it only *classifies* each model's own response (refused vs
not) and writes **aggregate rates only** — never the raw (potentially harmful) generations.

Env: NINEROUTER_BASE_URL, NINEROUTER_API_KEY.
Usage:
  python scripts/safety_leaderboard.py \
    --models cx/gpt-5.4-mini,cc/claude-haiku-4-5-20251001,cx/gpt-5.5,cc/claude-sonnet-4-6 \
    --per-domain 40 --out results
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vibench.dataset import DEFAULT_DATA_PATH, Track, classify_track, load_dataset  # noqa: E402
from vibench.models import NineRouterModel  # noqa: E402
from vibench.safety import RuleJudge, evaluate_safety  # noqa: E402


def sample(rows, per_domain):
    seen: dict[str, int] = defaultdict(int)
    out = []
    for r in rows:
        if classify_track(r) != Track.GENERATION or r.domain not in ("hatespeech", "politics_easy", "politics_advanced"):
            continue
        if seen[r.domain] < per_domain:
            out.append(r)
            seen[r.domain] += 1
    return out


def gen_model(mid, rows, workers):
    m = NineRouterModel(model=mid)

    def one(r):
        try:
            return r.id, m.generate(r)
        except Exception:
            return r.id, ""

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return {rid: resp for rid, resp in ex.map(one, rows)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--per-domain", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = sample(load_dataset(args.data), args.per_domain)
    print(f"{len(rows)} safety prompts x {len(models)} models", file=sys.stderr)

    report = {"models": models, "n_prompts": len(rows), "per_model": {}}
    for mid in models:
        print(f"generating {mid} …", file=sys.stderr)
        resp = gen_model(mid, rows, args.workers)
        report["per_model"][mid] = evaluate_safety(rows, resp, judge=RuleJudge()).as_dict()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "safety.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Safety leaderboard — the track V-Bench marks *Inactive*",
        "",
        f"{len(rows)} stratified Safety prompts (hate-speech + politics) via a 9router gateway. "
        "`hate_safe_rate` = fraction of hate-speech prompts the model refused/deflected "
        "(**higher = safer**); `pol_answer_rate` = politics prompts engaged (not refused). "
        "Rule-based refusal judge; **aggregate rates only — no raw generations are stored**.",
        "",
        "| Model | hate n | hate_safe_rate | politics n | pol_answer_rate |",
        "|---|---|---|---|---|",
    ]
    for m in models:
        d = report["per_model"][m]
        hs, pol = d.get("hatespeech", {}), d.get("politics", {})
        lines.append(
            f"| {m.split('/')[-1]} | {hs.get('n', '-')} | {hs.get('safe_rate', '-')} "
            f"| {pol.get('n', '-')} | {pol.get('answer_rate', '-')} |"
        )
    (out / "SAFETY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
