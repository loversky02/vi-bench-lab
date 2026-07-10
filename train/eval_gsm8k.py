#!/usr/bin/env python3
"""Exact-match accuracy on the held-out Vi-GSM8K eval set (ground-truthed by
`final_answer`). This is the reasoning-lift metric: run it on the base model, then on the
GRPO adapter, and compare. GPU-gated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from reward import _norm_num, extract_final_answer
except ImportError:
    from train.reward import _norm_num, extract_final_answer


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter dir")
    ap.add_argument("--data", default="train/data/gsm8k_eval.jsonl")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()][: args.limit]
    correct = 0
    for r in rows:
        text = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=args.max_new, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        correct += int(extract_final_answer(gen) == _norm_num(r["gold"]))

    acc = correct / len(rows) if rows else 0.0
    result = {
        "model": args.model, "adapter": args.adapter,
        "n": len(rows), "correct": correct, "accuracy": round(acc, 4),
    }
    print(json.dumps(result, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
