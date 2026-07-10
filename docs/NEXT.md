# NEXT — the live / paid roadmap

Everything in the repo is offline-verified $0 (17 tests, deterministic baseline on the
real 9,141-row set). What remains needs either a live LLM endpoint (9router, ~free for
GPT-5.5/Claude) or metered GPU (the training lever). Ordered by leverage.

## A. Live baselines (9router, ~$0)
- [ ] Set `NINEROUTER_BASE_URL` / `NINEROUTER_API_KEY`.
- [ ] `run --model 9router --model-id gpt-5.5` over the 5,141 scored rows → real submission.
- [ ] Repeat for a few open models (Qwen3-4B/32B via the gateway) to get a local baseline table.

## B. Silver key + local iteration (9router, ~$0)
- [ ] `silver-key --ensemble "gpt-5.5,claude-sonnet-5,gemini-2.5-flash"` → `out/silver.jsonl`.
- [ ] `score` each candidate against the silver key; track per-category deltas.
- [ ] Reality check: submit 1–2 candidates to the real grader and measure silver↔official gap.

## C. Confirm the submission spec (email, $0)
- [ ] Verify the **agentic** answer JSON shape against the grader (object vs OpenAI
      tool-call). The MC spec (`{"id","answer":"<letter>"}`) is confirmed from the site.
- [ ] Contact `vbench-support@vinuni.edu.vn` for the professional test set / official submit.

## D. ⭐ The Vi-GSM8K lever (metered GPU — the paper)
The audit shows the field is weakest and *most separable* on reasoning: `Logic` (r=0.43
with total, mean ≈ 40), `Lý`, `Văn`, `Toán`. This is exactly what the
[[vi-gsm8k-agentic]] dataset targets.
- [ ] Baseline Qwen3-4B on the MC reasoning slice (logics/mathematics/physics/literature).
- [ ] SFT → GRPO on Vi-GSM8K-style VN reasoning data (RunPod; reuse the GRPO env runbook).
- [ ] Re-score the reasoning slice against the silver key; measure the lift.
- [ ] Submit → land a small open model on the national leaderboard.
- [ ] Optional tie-in: **Vi-Calibrate** (faithful-confidence) — routing on V-Bench MC.

## E. Publish the Safety leaderboard (9router judge, ~$0)
- [ ] `safety --model <each> --judge 9router` over hatespeech (refusal) + politics (stance).
- [ ] Rank models by `safe_rate`; this is the view V-Bench marks Inactive.

## F. Ship
- [ ] `git init` + push to `github.com/loversky02/vi-bench-lab` (public).
- [ ] Per repo policy: **omit** `Co-Authored-By` AI trailers on commits (academic repo).
- [ ] Feature on the profile pins + README once a model lands on the board.
