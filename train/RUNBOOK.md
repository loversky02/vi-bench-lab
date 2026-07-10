# RUNBOOK — the reasoning lever (GRPO on Vi-GSM8K → V-Bench reasoning slice)

Offline-verified parts (reward, data bridge) run $0 on a Mac. Training is GPU-gated —
rent an hourly GPU (RunPod / FPT). Ordered.

## 0. Why this slice
V-Bench audit: `Logic` correlates only **r=0.43** with the total (nearly orthogonal),
`Lý`/`Toán`/`Văn` sit at the bottom (mean ≈ 40). Moving reasoning lifts the total
*without* chasing saturated categories (`Hate speech` ≈ 92). Verifiable reward → GRPO.

## 1. Build data ($0, local)
```bash
# Vi-GSM8K = vuongtsc/vi-gsm8k-agentic (HF). Download a jsonl, then:
python train/build_sft_data.py --in vi_gsm8k.jsonl --out train/data
# -> train/data/sft.jsonl (chat SFT) + train/data/grpo.jsonl ({prompt, gold})
```

## 2. Baseline the slice BEFORE training ($0 via 9router, or GPU for a local ckpt)
```bash
export NINEROUTER_BASE_URL=... NINEROUTER_API_KEY=...
python -m vibench run --model 9router --model-id cx/gpt-5.5 \
  --domains logics,mathematics,physics,literature --per-domain 40 --out out/base_gpt55.jsonl
python scripts/live_baseline.py --models "<baseline models>" \
  --domains logics,mathematics,physics,literature --per-domain 40
```

## 3. GPU env (RunPod) — pinned to avoid the vllm/torch/cuda mismatch
```bash
# A5000/A100. Pin transformers 4.51 + trl 0.17 (newer trl breaks GRPOConfig args).
pip install "transformers==4.51.*" "trl==0.17.*" datasets accelerate vllm
# optional SFT warm-start first (train/data/sft.jsonl), then GRPO:
setsid python train/train_grpo.py --model Qwen/Qwen3-4B \
  --data train/data/grpo.jsonl --out train/out/grpo-qwen3-4b --max-steps 500 \
  > train.log 2>&1 &
# kill a stuck run without killing the launcher:  pkill -f '[t]rain_grpo'
```

## 4. Re-score the slice, measure the lift, submit
```bash
# serve the trained ckpt (vllm, OpenAI-compatible) and point the harness at it:
python -m vibench run --model 9router --model-id qwen3-4b-grpo \
  --domains logics,mathematics,physics,literature --per-domain 40 --out out/grpo.jsonl
# then run the FULL scored set and email the submission to vbench-support@vinuni.edu.vn
```

## 5. Teardown
Stop the pod immediately after copying `train/out/` + logs (GPU is metered = real money).

## Honest expectations
GRPO lifts a *specific* verifiable slice; it is not a general capability jump. Report the
per-category delta on logics/math/physics honestly, including where it does **not** move
(e.g. `Văn` is not numerically verifiable — SFT-only there). Silver-key scores are a
proxy; the official number comes from the V-Bench grader.
