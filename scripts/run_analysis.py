"""Command-line analysis + PDF report (useful for batch testing and sample outputs).

Examples:
  python scripts/run_analysis.py --current data/sample_policies/current_policy_nilgiri_arogya_basic.pdf \
      --proposed data/sample_policies/proposed_policy_nilgiri_arogya_plus.pdf --age 34 --planning-child
  python scripts/run_analysis.py --current my_policy.pdf --offline
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from coverwise import config  # noqa: E402
from coverwise.llm import LLMClient  # noqa: E402
from coverwise.pipeline import analyze_pair  # noqa: E402
from coverwise.qa import answer_question  # noqa: E402
from coverwise.report import generate_report  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", required=True)
    ap.add_argument("--proposed")
    ap.add_argument("--age", type=int, default=35)
    ap.add_argument("--family", default="Self, spouse, 1 child")
    ap.add_argument("--has-ped", action="store_true")
    ap.add_argument("--planning-child", action="store_true")
    ap.add_argument("--current-premium", type=int)
    ap.add_argument("--proposed-premium", type=int)
    ap.add_argument("--current-si", type=int)
    ap.add_argument("--proposed-si", type=int)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out", default=str(config.REPORTS_DIR / "coverwise_report.pdf"))
    a = ap.parse_args()

    llm = LLMClient("none") if a.offline else LLMClient()
    profile = {"age": a.age, "family_members": a.family, "has_ped": a.has_ped,
               "planning_child": a.planning_child}
    res = analyze_pair(a.current, a.proposed, llm, profile,
                       {"annual_premium": a.current_premium, "sum_insured": a.current_si},
                       {"annual_premium": a.proposed_premium, "sum_insured": a.proposed_si},
                       progress=lambda f, m: print(f"[{f:4.0%}] {m}"))
    qa_log = []
    target = res["proposed"] or res["current"]
    for q in ["Is co-payment applicable in this policy?", "Is room rent capped?",
              "What is the waiting period for pre-existing diseases?", "Should I switch to this policy?"]:
        r = answer_question(target["index"], q, llm, profile, analysis=target)
        qa_log.append({"question": q, "answer": r["answer"]})
    Path(a.out).write_bytes(generate_report(res, profile, qa_log))
    print(f"\nLLM: {llm.label}")
    for key in ("current", "proposed"):
        if res[key]:
            s = res[key]["score"]
            print(f"{key:9s}: {s['score']}/100 {s['label_display']} | flags {res[key]['features']['risk_flags_count']}")
    if res["comparison"]:
        print("Improved:", res["comparison"]["improved"])
        print("Worse   :", res["comparison"]["worse"])
    print("Report  :", a.out)
