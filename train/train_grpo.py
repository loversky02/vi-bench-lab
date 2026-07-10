"""GRPO on the reasoning slice (GPU-gated; runs on RunPod, not on the Mac).

Reward = train/reward.py (verifiable exact-match on the final answer + format bonus).
Data  = train/data/grpo.jsonl from build_sft_data.py ({"prompt","gold"}).

Kept import-safe: heavy deps (torch/trl/transformers) are imported inside main() so the
module can be imported / linted without a GPU. See RUNBOOK.md for the pinned env.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--data", default="train/data/grpo.jsonl")
    ap.add_argument("--out", default="train/out/grpo-qwen3-4b")
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--max-steps", type=int, default=500)
    args = ap.parse_args(argv)

    # Heavy imports are deliberately local (GPU-only).
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    from reward import trl_reward_func  # noqa: F401  (run from train/)

    rows = load_jsonl(args.data)
    ds = Dataset.from_list([{"prompt": r["prompt"], "gold": r["gold"]} for r in rows])

    cfg = GRPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        gradient_checkpointing=True,
    )
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[trl_reward_func],
        args=cfg,
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
