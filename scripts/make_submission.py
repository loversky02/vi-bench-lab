#!/usr/bin/env python3
"""Generate a V-Bench submission.jsonl by running a model over the scored tracks (MC +
agentic) via an OpenAI-compatible endpoint (e.g. a local `vllm serve`). Threaded for
throughput; reuses the tested harness prompt/parse + the official `vbench` agentic format.

Usage (on a pod with `vllm serve /root/sft-merged --port 8000` running):
  python scripts/make_submission.py --model /root/sft-merged --base-url http://localhost:8000/v1 \
    --out out/submission.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vibench.dataset import DEFAULT_DATA_PATH, SCORED_TRACKS, classify_track, load_dataset  # noqa: E402
from vibench.harness import predict_row  # noqa: E402
from vibench.models import NineRouterModel  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    ap.add_argument("--out", default="out/submission.jsonl")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    rows = [r for r in load_dataset(args.data) if classify_track(r) in SCORED_TRACKS]
    if args.limit:
        rows = rows[: args.limit]
    m = NineRouterModel(model=args.model, base_url=args.base_url, api_key="local")
    print(f"{len(rows)} scored rows (MC + agentic) -> {args.out}", file=sys.stderr)

    def one(r):
        try:
            return {"id": r.id, "answer": predict_row(m, r, agentic_format="vbench")}
        except Exception:
            return None

    subs, n = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for s in ex.map(one, rows):
            n += 1
            if s is not None:
                subs.append(s)
            if n % 500 == 0:
                print(f"  {n}/{len(rows)}", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for s in subs:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"wrote {len(subs)}/{len(rows)} answers -> {out}")


if __name__ == "__main__":
    main()
