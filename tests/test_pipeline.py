"""Offline tests: python -m pytest -q   (no API key needed).

A FakeLLM exercises every LLM code path (extraction merge, summary, comparison,
advisor questions, RAG answers) without network calls.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coverwise import config  # noqa: E402
from coverwise.analysis import is_advice_question, sanitize  # noqa: E402
from coverwise.llm import LLMClient, parse_json  # noqa: E402
from coverwise.pipeline import analyze_pair  # noqa: E402
from coverwise.qa import answer_question  # noqa: E402
from coverwise.report import generate_report  # noqa: E402
from coverwise.scoring import rule_based_score, score_policy  # noqa: E402
from coverwise.text_clean import anonymize, split_sentences  # noqa: E402

CUR = config.SAMPLE_POLICY_DIR / "current_policy_nilgiri_arogya_basic.pdf"
PROP = config.SAMPLE_POLICY_DIR / "proposed_policy_nilgiri_arogya_plus.pdf"


class FakeLLM(LLMClient):
    def __init__(self):
        super().__init__("none")
        self.provider, self.model, self.calls = "fake", "fake-1", 0

    @property
    def available(self):
        return True

    def chat(self, system, user, json_mode=False, max_tokens=2048, retries=1):
        self.calls += 1
        if json_mode:
            if '"questions"' in user:
                return json.dumps({"questions": [f"Question {i} about co-payment?" for i in range(8)]})
            return json.dumps({"copayment_percentage": "10%", "ped_waiting_period_months": "4 years",
                               "restoration_benefit": "no", "evidence": {"copayment_percentage":
                                                                        {"page": 2, "quote": "co-payment of 10%"}},
                               "ambiguous_clauses": ["at its sole discretion"]})
        if "summary" in user.lower() and "10 to 15" in user:
            return "\n".join(f"- line {i}" for i in range(10))
        return "Yes, a 10% co-payment applies [S1].\nVerify: confirm with insurer. You should buy this policy."


def test_utils():
    assert split_sentences("Pay Rs. 5,000 per day. Next one.") == ["Pay Rs. 5,000 per day.", "Next one."]
    assert "[PHONE]" in anonymize("call 9876543210") and "[PAN]" in anonymize("PAN ABCDE1234F")
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert is_advice_question("Should I buy this policy?")
    assert "you should buy" not in sanitize("You should buy this policy").lower()


def test_scoring_bounds():
    f = {k: 0 for k in config.MODEL_FEATURES}
    f.update(policy_type="Family Floater", sum_insured=500000, annual_premium=10000,
             claim_process_clarity_score=100, exclusion_count=5, premium_to_coverage_ratio=0.02)
    r = rule_based_score(f)
    assert 0 <= r["score"] <= 100
    s = score_policy(f)
    assert s["label"] in config.RISK_LABELS


def test_offline_pipeline_and_report():
    res = analyze_pair(CUR, PROP, LLMClient("none"), {"age": 34, "planning_child": True})
    assert res["current"]["extraction"]["copayment_percentage"] == 10
    assert res["proposed"]["extraction"]["restoration_benefit"] is True
    assert res["proposed"]["score"]["score"] > res["current"]["score"]["score"]
    ans = answer_question(res["current"]["index"], "Is room rent capped?")
    assert "[S" in ans["answer"]
    pdf = generate_report(res, {"age": 34})
    assert pdf[:5] == b"%PDF-" and len(pdf) > 5000


def test_llm_paths_with_fake():
    llm = FakeLLM()
    res = analyze_pair(CUR, PROP, llm, {"age": 62})
    ext = res["current"]["extraction"]
    assert ext["extraction_method"] == "llm+rules"
    assert ext["ped_waiting_period_months"] == 48  # "4 years" coerced to months
    assert res["comparison"]["narrative"]
    assert len(res["proposed"]["advisor_questions"]) >= 6
    ans = answer_question(res["current"]["index"], "Is co-payment applicable?", llm)
    assert "you should buy" not in ans["answer"].lower()
    assert generate_report(res, {"age": 62}, [{"question": "q", "answer": ans["answer"]}])[:5] == b"%PDF-"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
