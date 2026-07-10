from pathlib import Path

from vibench.audit import (
    audit_report,
    category_stats,
    load_leaderboard,
    model_gaps,
    saturation,
)
from vibench.dataset import load_dataset

FIX = Path(__file__).parent / "fixtures" / "mini.jsonl"


def test_load_leaderboard():
    lb = load_leaderboard()
    assert len(lb["models"]) == 10
    assert len(lb["categories"]) == 16
    assert lb["models"][0]["name"] == "Gemini-3.5-Flash"
    assert lb["models"][0]["scores"]["Toán"] == 83.2


def test_category_stats():
    stats = category_stats(load_leaderboard())
    assert set(stats) == set(load_leaderboard()["categories"])
    for s in stats.values():
        assert s["min"] <= s["mean"] <= s["max"]


def test_saturation_and_gaps():
    lb = load_leaderboard()
    sat = saturation(lb)
    assert "Hate speech" in sat["saturated"]
    assert "Logic" in sat["hard"]
    gaps = model_gaps(lb)
    assert gaps[0]["name"] == "V-LLM v1"  # biggest sovereignty-minus-reasoning spike
    assert gaps[0]["sovereignty_minus_reasoning"] > 40


def test_audit_report_with_public_set():
    rep = audit_report(load_leaderboard(), rows=load_dataset(FIX))
    assert rep["findings"]
    assert rep["public_set"]["total"] == 5
    assert rep["public_set"]["agentic_avg_candidates"] == 1
