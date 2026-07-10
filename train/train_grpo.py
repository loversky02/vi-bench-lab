"""GRPO on the reasoning slice (GPU-gated; runs on RunPod, not on the Mac).

Reward = train/reward.py (verifiable exact-match on the final answer + format bonus).
Data  = train/data/grpo_train.jsonl from build_sft_data.py
        ({"prompt": [chat messages], "gold": <final answer>}).

LoRA by default so a 3–4B model fits a 24GB card. Heavy deps import inside main() so the
module stays import-safe without a GPU. Pinned env: torch 2.4+ · transformers 4.51.3 ·
trl 0.17.0 · datasets 3.5.0 · peft (NO vllm — it drags in a torch that breaks CUDA).
"""

from __future__ import annotations

import argparse


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--data", default="train/data/grpo_train.jsonl")
    ap.add_argument("--out", default="train/out/grpo")
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--max-prompt", type=int, default=400)
    ap.add_argument("--max-completion", type=int, default=256)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--vllm", action="store_true",
                    help="fast rollout generation via vLLM (needs an aligned vLLM install; see train/VLLM.md)")
    args = ap.parse_args(argv)

    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    try:
        from reward import trl_reward_func
    except ImportError:
        from train.reward import trl_reward_func

    ds = load_dataset("json", data_files=args.data, split="train")

    peft_config = None
    if not args.no_lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules="all-linear", task_type="CAUSAL_LM",
        )

    cfg = GRPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt,
        max_completion_length=args.max_completion,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        logging_steps=5,
        save_steps=args.save_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # else LoRA grads are None
        use_vllm=args.vllm,  # ~5-10x faster rollouts when vLLM is installed & CUDA-aligned
        report_to="none",
        log_completions=False,
    )
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[trl_reward_func],
        args=cfg,
        train_dataset=ds,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
