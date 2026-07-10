from pathlib import Path

from vibench.dataset import Track, load_dataset
from vibench.harness import build_submission
from vibench.models import DeterministicModel
from vibench.scorer import build_silver_key, score

FIX = Path(__file__).parent / "fixtures" / "mini.jsonl"


def test_score_counts():
    rows = load_dataset(FIX)
    submission = [{"id": 101, "answer": "B"}, {"id": 102, "answer": "A"}]
    key = {101: "B", 102: "C"}  # 101 correct, 102 wrong
    rep = score(submission, key, rows)
    assert rep.total_scored == 2
    assert rep.total_correct == 1
    assert rep.accuracy_micro == 0.5
    assert rep.by_category["Toán"]["acc"] == 1.0
    assert rep.by_category["Phương ngữ"]["acc"] == 0.0


def test_silver_key_self_consistency():
    rows = load_dataset(FIX)
    model = DeterministicModel()
    key = build_silver_key(rows, [model, model], track=Track.MC)
    assert set(key) == {101, 102}  # MC ids only
    subs, _ = build_submission(rows, model, tracks={Track.MC})
    rep = score(subs, key, rows)
    assert rep.total_scored == 2
    assert rep.total_correct == 2  # a model scored against its own silver key -> perfect
    assert rep.accuracy_micro == 1.0
