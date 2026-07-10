#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model → a full model on disk.

Used to turn an SFT LoRA (e.g. vuongtsc/qwen3-4b-vi-gsm8k-agentic on
Qwen/Qwen3-4B-Instruct-2507) into the warm-start we then run GRPO on top of. GPU-gated.
"""
from __future__ import annotations

import argparse


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # tokenizer from the BASE (the adapter's tokenizer_config can carry a malformed
    # extra_special_tokens list; the base vocab/chat-template is what SFT used anyway)
    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"merged {args.base} + {args.adapter} -> {args.out}")


if __name__ == "__main__":
    main()
