"""Clause-extraction accuracy against ground truth (evaluation metrics 1 & 4).

Usage: python scripts/evaluate_extraction.py [--offline]
Uses data/eval/sample_policy_ground_truth.json - add entries for your own PDFs
(file name -> expected values) to evaluate on real policies.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

from coverwise import config  # noqa: E402
from coverwise.llm import LLMClient  # noqa: E402
from coverwise.pipeline import analyze_policy  # noqa: E402


def compare(expected, got):
    if expected is None:
        return None
    if isinstance(expected, bool):
        return got is expected
    if isinstance(expected, (int, float)):
        return got is not None and abs(float(got) - float(expected)) < 1e-6
    return str(got).strip().upper() == str(expected).strip().upper()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="rules only, no LLM")
    args = ap.parse_args()
    llm = LLMClient("none") if args.offline else LLMClient()
    truth = json.loads((config.EVAL_DIR / "sample_policy_ground_truth.json").read_text())
    rows = []
    for fname, expected in truth.items():
        pdf = config.SAMPLE_POLICY_DIR / fname
        if not pdf.exists():
            print(f"skip {fname}: copy it into data/sample_policies/ to evaluate it")
            continue
        a = analyze_policy(pdf, fname, llm)
        ext, feat = a["extraction"], a["features"]
        for field, exp in expected.items():
            if field.startswith("_"):
                continue
            got = feat.get(field) if field in ("disease_sub_limit_count",) else ext.get(field)
            rows.append({"policy": fname, "field": field, "expected": exp, "extracted": got,
                         "correct": compare(exp, got)})
        print(f"{fname}: score {a['score']['score']} ({a['score']['label']}), {a['processing_seconds']}s")
    df = pd.DataFrame(rows)
    out = config.REPORTS_DIR / f"extraction_eval_{'offline' if args.offline else llm.provider}.csv"
    df.to_csv(out, index=False)
    print(df[~df.correct].to_string(index=False) if (~df.correct).any() else "All fields correct")
    print(f"\nField accuracy ({llm.label}): {df.correct.mean():.1%}  ({df.correct.sum()}/{len(df)})")
    print("saved", out)
