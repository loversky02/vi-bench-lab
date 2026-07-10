from pathlib import Path

from vibench.dataset import load_dataset
from vibench.models import DeterministicModel
from vibench.safety import RuleJudge, evaluate_safety, is_refusal

FIX = Path(__file__).parent / "fixtures" / "mini.jsonl"


def test_is_refusal():
    assert is_refusal("Xin lỗi, tôi không thể hỗ trợ.") is True
    assert is_refusal("Đây là nội dung bạn yêu cầu ...") is False
    assert is_refusal("") is False


def test_evaluate_safety_rule():
    rows = load_dataset(FIX)  # includes 104 hatespeech, 105 politics
    responses = {
        104: "Tôi không thể hỗ trợ yêu cầu này.",
        105: "Việt Nam khẳng định chủ quyền với quần đảo Hoàng Sa.",
    }
    rep = evaluate_safety(rows, responses, judge=RuleJudge())
    assert rep.hatespeech["n"] == 1
    assert rep.hatespeech["safe_rate"] == 1.0
    assert rep.politics["n"] == 1
    assert rep.politics["answer_rate"] == 1.0
    assert rep.politics["factuality"] is None  # rule judge cannot score factuality


def test_deterministic_refuser():
    rows = [r for r in load_dataset(FIX) if r.domain == "hatespeech"]
    model = DeterministicModel(refusal_rate=1.0)
    responses = {r.id: model.generate(r) for r in rows}
    rep = evaluate_safety(rows, responses)
    assert rep.hatespeech["safe_rate"] == 1.0
