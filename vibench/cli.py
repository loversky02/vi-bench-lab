"""Command-line interface: ``python -m vibench <command>``.

Commands
--------
  fetch        download the public test set into data/
  stats        dataset breakdown by track / domain
  run          run a model over scored tracks -> submission.jsonl        (#1)
  silver-key   build a majority-vote silver key from an ensemble          (#1)
  score        score a submission against an answer/silver key            (#1)
  safety       score the Safety track (refusal + stance)                  (#2)
  audit        meta-analysis of the leaderboard + public set              (#3)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from . import audit as audit_mod
from .dataset import DEFAULT_DATA_PATH, Track, classify_track, load_dataset
from .harness import build_submission, write_submission
from .models import DeterministicModel, get_model
from .safety import NineRouterJudge, RuleJudge, evaluate_safety
from .scorer import build_silver_key, load_key, save_key, score

PUBLIC_TEST_URL = "https://vbench.ai/downloads/public-test.jsonl"
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _progress(i: int, n: int) -> None:
    print(f"  … {i}/{n}", file=sys.stderr)


# --------------------------------------------------------------------------- #
def cmd_fetch(args) -> int:
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(PUBLIC_TEST_URL, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())
    n = sum(1 for _ in open(dest, encoding="utf-8"))
    print(f"saved {n} rows -> {dest}")
    return 0


def cmd_stats(args) -> int:
    rows = load_dataset(args.data)
    by_track: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    for r in rows:
        by_track[classify_track(r).value] = by_track.get(classify_track(r).value, 0) + 1
        by_domain[r.domain] = by_domain.get(r.domain, 0) + 1
    _print(
        {
            "total": len(rows),
            "by_track": by_track,
            "by_domain": dict(sorted(by_domain.items(), key=lambda kv: -kv[1])),
        }
    )
    return 0


def _make_model(args):
    kw = {}
    if args.model != "deterministic" and getattr(args, "model_id", None):
        kw["model"] = args.model_id
    return get_model(args.model, **kw)


def _filter_rows(rows, domains=None, per_domain=None):
    if domains:
        wanted = {d.strip() for d in domains.split(",") if d.strip()}
        rows = [r for r in rows if r.domain in wanted]
    if per_domain:
        from collections import defaultdict
        seen: dict[str, int] = defaultdict(int)
        out = []
        for r in rows:
            if seen[r.domain] < per_domain:
                out.append(r)
                seen[r.domain] += 1
        rows = out
    return rows


def cmd_run(args) -> int:
    rows = _filter_rows(load_dataset(args.data), args.domains, args.per_domain)
    model = _make_model(args)
    tracks = {Track.MC, Track.AGENTIC} if args.tracks == "scored" else {Track(args.tracks)}
    subs, stats = build_submission(
        rows, model, tracks=tracks, limit=args.limit,
        agentic_format=args.agentic_format, progress=_progress,
    )
    out = write_submission(subs, args.out)
    print(f"wrote {len(subs)} answers -> {out}")
    _print(stats.as_dict())
    return 0


def cmd_silver_key(args) -> int:
    rows = load_dataset(args.data)
    if args.offline:
        models = [DeterministicModel()]
    else:
        ids = [s.strip() for s in args.ensemble.split(",") if s.strip()]
        models = [get_model("9router", model=mid) for mid in ids]
    key = build_silver_key(rows, models, track=Track.MC)
    if args.limit:
        key = dict(list(key.items())[: args.limit])
    save_key(key, args.out)
    print(f"silver key ({len(key)} MC ids, {len(models)} voter(s)) -> {args.out}")
    return 0


def cmd_score(args) -> int:
    rows = load_dataset(args.data)
    submission = [json.loads(l) for l in open(args.submission, encoding="utf-8") if l.strip()]
    key = load_key(args.key)
    report = score(submission, key, rows)
    _print(report.as_dict())
    return 0


def cmd_safety(args) -> int:
    rows = [r for r in load_dataset(args.data) if classify_track(r) == Track.GENERATION]
    if args.limit:
        rows = rows[: args.limit]
    model = _make_model(args)
    responses = {r.id: model.generate(r) for r in rows}
    judge = NineRouterJudge() if args.judge == "9router" else RuleJudge()
    report = evaluate_safety(rows, responses, judge=judge)
    _print(report.as_dict())
    return 0


def cmd_audit(args) -> int:
    lb = audit_mod.load_leaderboard(args.leaderboard) if args.leaderboard else audit_mod.load_leaderboard()
    rows = None
    if args.data and Path(args.data).exists():
        rows = load_dataset(args.data)
    report = audit_mod.audit_report(lb, rows=rows)
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"audit report -> {args.out}")
    print("\n".join(f"• {f}" for f in report["findings"]))
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vibench", description="Open $0 toolkit around V-Bench (vbench.ai)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("fetch", help="download the public test set")
    sp.add_argument("--out", default=str(DEFAULT_DATA_PATH))
    sp.set_defaults(func=cmd_fetch)

    sp = sub.add_parser("stats", help="dataset breakdown")
    sp.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    sp.set_defaults(func=cmd_stats)

    sp = sub.add_parser("run", help="#1 build submission.jsonl")
    sp.add_argument("--model", default="deterministic", help="backend: deterministic | 9router")
    sp.add_argument("--model-id", default=None, help="LLM id for the 9router backend")
    sp.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    sp.add_argument("--out", default="out/submission.jsonl")
    sp.add_argument("--tracks", default="scored", help="scored | mc | agentic")
    sp.add_argument("--agentic-format", default="object", choices=["object", "openai"])
    sp.add_argument("--domains", default=None, help="comma-separated domains to include")
    sp.add_argument("--per-domain", type=int, default=None, help="cap rows sampled per domain (stratified)")
    sp.add_argument("--limit", type=int, default=None)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("silver-key", help="#1 majority-vote silver key")
    sp.add_argument("--ensemble", default="gpt-5.5,claude-sonnet-5,gemini-2.5-flash",
                    help="comma-separated 9router model ids")
    sp.add_argument("--offline", action="store_true", help="use a single DeterministicModel (test only)")
    sp.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    sp.add_argument("--out", default="out/silver-key.jsonl")
    sp.add_argument("--limit", type=int, default=None)
    sp.set_defaults(func=cmd_silver_key)

    sp = sub.add_parser("score", help="#1 score a submission against a key")
    sp.add_argument("--submission", required=True)
    sp.add_argument("--key", required=True)
    sp.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("safety", help="#2 score the Safety track")
    sp.add_argument("--model", default="deterministic")
    sp.add_argument("--model-id", default=None)
    sp.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    sp.add_argument("--judge", default="rule", choices=["rule", "9router"])
    sp.add_argument("--limit", type=int, default=None)
    sp.set_defaults(func=cmd_safety)

    sp = sub.add_parser("audit", help="#3 leaderboard + public-set meta-analysis")
    sp.add_argument("--leaderboard", default=None)
    sp.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_audit)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
