import json
import pathlib

from train.build_sft_data import convert
from train.reward import (
    answer_reward,
    extract_final_answer,
    format_reward,
    reasoning_reward,
    trl_reward_func,
)

FIX = pathlib.Path(__file__).parent / "fixtures" / "vi_gsm8k_mini.jsonl"


def test_extract_final_answer():
    assert extract_final_answer("suy luận... #### 8") == 8
    assert extract_final_answer("kết quả là \\boxed{42}") == 42
    assert extract_final_answer("tổng cộng 3 + 5 = 8 quả") == 8  # last number fallback
    assert extract_final_answer("20000 đồng") == 20000
    assert extract_final_answer("") is None


def test_rewards():
    assert answer_reward("#### 8", 8) == 1.0
    assert answer_reward("#### 7", 8) == 0.0
    assert format_reward("#### 8") == 0.1
    assert format_reward("chỉ có 8") == 0.0
    assert reasoning_reward("work... #### 8", "8") == 1.1
    assert reasoning_reward("8 không có hash", 8) == 1.0  # correct, no format bonus


def test_trl_reward_func():
    comps = ["#### 8", [{"role": "assistant", "content": "#### 7"}]]
    assert trl_reward_func(comps, gold=[8, 8]) == [1.1, 0.1]


def test_convert(tmp_path):
    stats = convert(FIX, tmp_path)
    assert stats == {"sft": 3, "grpo": 3, "skipped": 0}
    grpo = [json.loads(l) for l in open(tmp_path / "grpo.jsonl", encoding="utf-8")]
    assert grpo[0]["gold"] == 8 and grpo[2]["gold"] == 20000
    sft = [json.loads(l) for l in open(tmp_path / "sft.jsonl", encoding="utf-8")]
    assert sft[0]["messages"][1]["role"] == "user"
