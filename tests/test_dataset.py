from pathlib import Path

from vibench.dataset import Track, choice_letters, classify_track, load_dataset

FIX = Path(__file__).parent / "fixtures" / "mini.jsonl"


def test_load_and_tracks():
    rows = load_dataset(FIX)
    assert len(rows) == 5
    by_id = {r.id: r for r in rows}
    assert classify_track(by_id[101]) == Track.MC
    assert classify_track(by_id[102]) == Track.MC
    assert classify_track(by_id[103]) == Track.AGENTIC
    assert classify_track(by_id[104]) == Track.GENERATION
    assert classify_track(by_id[105]) == Track.GENERATION


def test_choice_letters():
    assert choice_letters(["A. x", "B. y", "C. z", "D. w"]) == ["A", "B", "C", "D"]
    assert choice_letters(["A. x", "B. y", "C. z"]) == ["A", "B", "C"]
    assert choice_letters(["x", "y"]) == ["A", "B"]  # no prefixes -> index fallback


def test_taxonomy():
    by_id = {r.id: r for r in load_dataset(FIX)}
    assert by_id[101].category == "Toán"
    assert by_id[101].group == "Kiến thức học thuật"
    assert by_id[103].category == "Agentic"
    assert by_id[104].category == "Hate speech"
    assert by_id[104].group == "An toàn và chủ quyền số"
