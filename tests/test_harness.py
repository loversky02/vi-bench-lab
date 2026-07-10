import json
from pathlib import Path

from vibench.dataset import Track, load_dataset
from vibench.harness import build_submission, predict_row, write_submission
from vibench.models import DeterministicModel

FIX = Path(__file__).parent / "fixtures" / "mini.jsonl"


def test_build_submission_scored_only():
    rows = load_dataset(FIX)
    subs, stats = build_submission(rows, DeterministicModel())
    assert stats.total == 3  # 2 MC + 1 agentic; 2 generation excluded
    assert {s["id"] for s in subs} == {101, 102, 103}
    assert stats.by_track == {"mc": 2, "agentic": 1}


def test_mc_answers_valid_letters():
    rows = {r.id: r for r in load_dataset(FIX)}
    model = DeterministicModel()
    assert predict_row(model, rows[101]) in ["A", "B", "C", "D"]
    assert predict_row(model, rows[102]) in ["A", "B", "C"]


def test_agentic_answer_shape():
    rows = {r.id: r for r in load_dataset(FIX)}
    model = DeterministicModel()
    # default = vbench official format: [ {"<fn_name>": {args}} ]
    ans = predict_row(model, rows[103])
    assert isinstance(ans, list) and len(ans) == 1
    call = ans[0]
    assert list(call.keys()) == ["calc_safe_velocity"]
    assert call["calc_safe_velocity"]["radius_m"] == 0     # number default
    assert call["calc_safe_velocity"]["surface"] == "DRY"  # first enum
    # object format still available
    assert predict_row(model, rows[103], agentic_format="object")["name"] == "calc_safe_velocity"
    # openai format: arguments as a JSON string
    ans2 = predict_row(model, rows[103], agentic_format="openai")
    assert isinstance(ans2["arguments"], str)
    assert json.loads(ans2["arguments"])["surface"] == "DRY"


def test_determinism():
    rows = load_dataset(FIX)
    m = DeterministicModel()
    assert build_submission(rows, m)[0] == build_submission(rows, m)[0]


def test_write_roundtrip(tmp_path):
    rows = load_dataset(FIX)
    subs, _ = build_submission(rows, DeterministicModel())
    p = write_submission(subs, tmp_path / "sub.jsonl")
    assert [json.loads(l) for l in open(p, encoding="utf-8")] == subs
