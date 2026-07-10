"""Verifiable reward for the reasoning lever (GSM8K-style, language-agnostic).

The V-Bench audit shows `Logic`/`Lý`/`Toán` are the field's weak, *most separable*
frontier (Logic correlates only r=0.43 with the total). A small Vietnamese model can
move that slice with GRPO on a **verifiable** reward: did it reach the right final
number? No learned reward model, no judge — just exact-match on the final answer, with
a small format bonus for showing work then emitting a `#### <answer>` line.

Pure stdlib, unit-tested offline.
"""

from __future__ import annotations

import re

_NUM = re.compile(r"-?\d[\d.,]*\d|-?\d")
_HASH = re.compile(r"####\s*(-?[\d.,]+)")
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")


def _norm_num(s: str):
    s = str(s).strip().replace(" ", "").rstrip(".,")
    s = s.replace(",", "")  # treat comma as a thousands separator
    try:
        f = float(s)
        return int(f) if f.is_integer() else round(f, 6)
    except ValueError:
        return s or None


def extract_final_answer(text: str):
    """Pull the final answer: prefer `#### X`, then `\\boxed{X}`, then the last number."""
    if not text:
        return None
    m = _HASH.search(text)
    if m:
        return _norm_num(m.group(1))
    m = _BOXED.search(text)
    if m:
        return _norm_num(m.group(1))
    nums = _NUM.findall(text)
    return _norm_num(nums[-1]) if nums else None


def answer_reward(completion: str, gold) -> float:
    return 1.0 if extract_final_answer(completion) == _norm_num(gold) else 0.0


def format_reward(completion: str) -> float:
    return 0.1 if _HASH.search(completion or "") else 0.0


def reasoning_reward(completion: str, gold) -> float:
    """Total shaped reward = correctness (1.0) + format bonus (0.1)."""
    return answer_reward(completion, gold) + format_reward(completion)


def trl_reward_func(completions, gold, **kwargs):
    """Adapter for TRL GRPOTrainer (`reward_funcs=[trl_reward_func]`).

    TRL passes `completions` (list) and forwards dataset columns as kwargs; we read the
    per-sample `gold` column. Handles chat-format completions too.
    """
    texts = [c[-1]["content"] if isinstance(c, list) else c for c in completions]
    return [reasoning_reward(t, g) for t, g in zip(texts, gold)]
