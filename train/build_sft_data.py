"""Bridge Vi-GSM8K (vuongtsc/vi-gsm8k-agentic) -> training data for the reasoning lever.

Produces two files from a Vi-GSM8K-style JSONL (fields flexible: question/prompt +
answer/solution, optionally final_answer):

* ``sft.jsonl``  — chat SFT: {"messages": [user question, assistant full solution]}
* ``grpo.jsonl`` — GRPO: {"prompt": question, "gold": <final numeric answer>}

The GRPO file is what the verifiable reward (train/reward.py) scores against. Stdlib
only; unit-tested offline with a fixture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from reward import extract_final_answer  # when run from inside train/
except ImportError:  # when imported as the `train` package (tests, repo root)
    from train.reward import extract_final_answer

_Q_KEYS = ("question", "prompt", "problem", "instruction")
_A_KEYS = ("answer", "solution", "response", "output", "rationale", "chain_of_thought")
_FINAL_KEYS = ("final_answer", "final", "gold", "label")

SYSTEM = "Bạn là trợ lý giải toán. Hãy suy luận từng bước rồi kết thúc bằng dòng '#### <đáp án>'."


def _first(o: dict, keys):
    for k in keys:
        if k in o and o[k] not in (None, ""):
            return o[k]
    return None


def to_records(o: dict):
    q = _first(o, _Q_KEYS)
    a = _first(o, _A_KEYS)
    if q is None:
        return None, None
    gold = _first(o, _FINAL_KEYS)
    if gold is None and a is not None:
        gold = extract_final_answer(str(a))
    a_text = str(a) if a is not None else ""
    if gold is not None and "####" not in a_text:
        a_text = f"{a_text}\n#### {gold}".strip()
    sft = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": str(q)},
            {"role": "assistant", "content": a_text},
        ]
    }
    # Conversational prompt so TRL applies the chat template (instruct models).
    grpo = {
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": str(q)},
        ],
        "gold": gold,
    }
    return sft, grpo


def convert(in_path: str | Path, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_sft = n_grpo = n_skip = 0
    with open(in_path, encoding="utf-8") as f, \
         open(out_dir / "sft.jsonl", "w", encoding="utf-8") as fs, \
         open(out_dir / "grpo.jsonl", "w", encoding="utf-8") as fg:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sft, grpo = to_records(json.loads(line))
            if sft is None:
                n_skip += 1
                continue
            fs.write(json.dumps(sft, ensure_ascii=False) + "\n")
            n_sft += 1
            if grpo["gold"] is not None:
                fg.write(json.dumps(grpo, ensure_ascii=False) + "\n")
                n_grpo += 1
    return {"sft": n_sft, "grpo": n_grpo, "skipped": n_skip}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Vi-GSM8K jsonl")
    ap.add_argument("--out", default="train/data")
    args = ap.parse_args()
    print(json.dumps(convert(args.inp, args.out)))
