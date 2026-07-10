#!/usr/bin/env python3
"""Live MC baseline over a stratified slice via an OpenAI-compatible gateway.

Runs several models over the same rows and reports, per domain:
  * consensus  — mean top-vote fraction across models (ground-truth-free signal:
                 high = models agree = easy/saturated; low = genuine disagreement)
  * agreement  — each model's agreement with the majority silver key

No ground truth is used: this is agreement analysis, not official scoring. Submit the
generated submissions to V-Bench for official numbers.

Env: NINEROUTER_BASE_URL, NINEROUTER_API_KEY (see README).
Usage:
  python scripts/live_baseline.py \
    --models cc/claude-haiku-4-5-20251001,cc/claude-sonnet-4-6,cc/claude-opus-4-8 \
    --domains logics,mathematics,physics,literature,dialect,culture,computer_science,medicine \
    --per-domain 12 --workers 8 --out results
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vibench.dataset import DEFAULT_DATA_PATH, Track, classify_track, load_dataset  # noqa: E402
from vibench.models import NineRouterModel  # noqa: E402


def stratified(rows, domains, per_domain):
    wanted = [d.strip() for d in domains.split(",") if d.strip()]
    seen: dict[str, int] = defaultdict(int)
    out = []
    for r in rows:
        if classify_track(r) != Track.MC or r.domain not in wanted:
            continue
        if seen[r.domain] < per_domain:
            out.append(r)
            seen[r.domain] += 1
    return out


def run_model(model_id, rows, workers):
    m = NineRouterModel(model=model_id)

    def one(r):
        try:
            return r.id, m.predict_mc(r)
        except Exception:
            return r.id, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return {rid: ans for rid, ans in ex.map(one, rows)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--domains", default="logics,mathematics,physics,literature,dialect,culture,computer_science,medicine")
    ap.add_argument("--per-domain", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = stratified(load_dataset(args.data), args.domains, args.per_domain)
    by_id = {r.id: r for r in rows}
    print(f"{len(rows)} rows x {len(models)} models = {len(rows) * len(models)} calls", file=sys.stderr)

    preds: dict[str, dict] = {}
    outdir = ROOT / "out"
    outdir.mkdir(exist_ok=True)
    for mid in models:
        print(f"running {mid} …", file=sys.stderr)
        preds[mid] = run_model(mid, rows, args.workers)
        with open(outdir / f"sub_{mid.replace('/', '_')}.jsonl", "w", encoding="utf-8") as f:
            for rid, ans in preds[mid].items():
                if ans is not None:
                    f.write(json.dumps({"id": rid, "answer": ans}, ensure_ascii=False) + "\n")

    # coverage guards against a silently-failing model poisoning the silver key
    coverage = {m: round(sum(1 for v in preds[m].values() if v) / len(rows), 3) for m in models}
    working = [m for m in models if coverage[m] >= 0.5]
    dropped = [m for m in models if m not in working]
    if dropped:
        print(f"WARNING: dropped low-coverage models {dropped} "
              f"(coverage={[coverage[m] for m in dropped]})", file=sys.stderr)

    # majority silver key + agreement + consensus (working models only)
    silver: dict[int, str] = {}
    consensus_by_domain: dict[str, list[float]] = defaultdict(list)
    for rid, row in by_id.items():
        votes = [preds[m][rid] for m in working if preds[m].get(rid)]
        if not votes:
            continue
        top, n = Counter(votes).most_common(1)[0]
        silver[rid] = top
        consensus_by_domain[row.domain].append(n / len(votes))

    domains = sorted({r.domain for r in rows})
    report = {
        "models": models, "working": working, "dropped": dropped,
        "coverage": coverage, "n_rows": len(rows), "per_domain": {}, "overall": {},
    }
    for d in domains:
        ids = [r.id for r in rows if r.domain == d]
        row_rep = {"n": len(ids), "consensus": round(sum(consensus_by_domain[d]) / len(ids), 3), "agreement": {}}
        for m in working:
            hits = sum(1 for i in ids if preds[m].get(i) and preds[m][i] == silver.get(i))
            row_rep["agreement"][m] = round(hits / len(ids), 3)
        report["per_domain"][d] = row_rep
    for m in working:
        hits = sum(1 for i in silver if preds[m].get(i) == silver[i])
        report["overall"][m] = round(hits / len(silver), 3) if silver else 0.0
    report["overall"]["consensus"] = round(
        sum(v for vs in consensus_by_domain.values() for v in vs) / len(silver), 3
    ) if silver else 0.0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # markdown table
    short = [m.split("/")[-1] for m in working]
    lines = [
        "# Live baseline — cross-model agreement (ground-truth-free difficulty probe)",
        "",
        f"{len(rows)} stratified MC rows, {len(working)} frontier models via a self-hosted "
        "9router gateway. `consensus` = mean top-vote fraction across models "
        "(**high = easy/agreed, low = genuinely hard/ambiguous**); `agreement` = each model vs the "
        "majority silver key. No ground truth is used — submit to V-Bench for official scores.",
        "",
        "Coverage (answered/total): "
        + ", ".join(f"{m.split('/')[-1]} {coverage[m]}" for m in models)
        + (f"  — dropped {[d.split('/')[-1] for d in dropped]}" if dropped else ""),
        "",
        "| Domain | n | consensus | " + " | ".join(short) + " |",
        "|---|---|---|" + "---|" * len(short),
    ]
    for d in domains:
        rr = report["per_domain"][d]
        lines.append(
            f"| {d} | {rr['n']} | {rr['consensus']} | "
            + " | ".join(f"{rr['agreement'][m]}" for m in working) + " |"
        )
    rr = report["overall"]
    lines.append(
        f"| **overall** | {len(rows)} | {rr['consensus']} | "
        + " | ".join(f"{rr[m]}" for m in working) + " |"
    )
    (out / "BASELINE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}/baseline.json + {out}/BASELINE.md", file=sys.stderr)


if __name__ == "__main__":
    main()
