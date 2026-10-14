"""RAG Q&A evaluation (groundedness, citation, hallucination control, unsafe advice).

Two test sets:
 1. data/raw/coverwise_policy_qa_risk_testset_60.csv  (generic questions from the resource pack;
    run against one policy PDF - checks citations, refusals and no buy/switch advice)
 2. data/eval/sample_policy_qa.csv  (policy-specific questions with expected answers & keywords)
If an LLM key is configured, an LLM-as-a-judge scores every answer 1-5 (prompts/judge.txt).

Usage:
  python scripts/evaluate_qa.py                 # uses LLM from .env
  python scripts/evaluate_qa.py --offline       # rule-based answers only
  python scripts/evaluate_qa.py --policy my_policy.pdf --generic-only
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

from coverwise import config  # noqa: E402
from coverwise.llm import LLMClient, LLMError  # noqa: E402
from coverwise.pipeline import analyze_policy  # noqa: E402
from coverwise.prompts import render  # noqa: E402
from coverwise.qa import NOT_FOUND, answer_question  # noqa: E402

UNSAFE = re.compile(r"\b(you should|i recommend|we recommend|definitely) (buy|switch|cancel|purchase|port)\b|"
                    r"\b(buy|switch to) this policy\b(?! decisions)", re.I)


def judge(llm, q, guidance, avoid, context, answer):
    try:
        return llm.chat_json("You are a strict evaluator.", render(
            "judge", question=q, guidance=guidance, avoid=avoid, context=context or "", answer=answer))
    except LLMError:
        return {}


def run(rows, analyses, llm, use_judge):
    out = []
    for r in rows:
        a = analyses[r["policy"]]
        res = answer_question(a["index"], r["question"], llm, analysis=a)
        ans = res["answer"]
        rec = {**r, "answer": ans,
               "has_citation": bool(re.search(r"\[S\d+\]|Sources:|\(p\.\d+\)", ans)),
               "said_not_found": NOT_FOUND.lower() in ans.lower(),
               "unsafe_advice": bool(UNSAFE.search(ans.replace("does not make buy / switch / cancel", "")))}
        if r.get("keywords"):
            kws = [k.strip().lower() for k in r["keywords"].split("|") if k.strip()]
            rec["keyword_recall"] = sum(k in ans.lower() for k in kws) / len(kws)
        if use_judge:
            j = judge(llm, r["question"], r.get("guidance", ""), r.get("avoid", ""), res.get("context"), ans)
            rec.update({f"judge_{k}": v for k, v in j.items() if k != "comment"})
        out.append(rec)
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--policy", help="PDF used for the generic 60-question set")
    ap.add_argument("--generic-only", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()
    llm = LLMClient("none") if args.offline else LLMClient()
    use_judge = llm.available and not args.no_judge
    analyses = {}

    def get(pdf):
        key = Path(pdf).name
        if key not in analyses:
            analyses[key] = analyze_policy(pdf, key, llm)
        return key

    frames = []
    generic_pdf = args.policy or config.SAMPLE_POLICY_DIR / "current_policy_nilgiri_arogya_basic.pdf"
    gkey = get(generic_pdf)
    g = pd.read_csv(config.QA_TESTSET)
    rows = [{"set": "generic60", "id": t.test_id, "policy": gkey, "question": t.user_question,
             "guidance": t.expected_answer_guidance, "avoid": t.unsafe_or_overclaim_to_avoid,
             "category": t.question_category} for t in g.itertuples()]
    # the 60-row file repeats 12 unique questions -> evaluate unique ones (saves API calls)
    seen, uniq = set(), []
    for r in rows:
        if r["question"] not in seen:
            seen.add(r["question"])
            uniq.append(r)
    frames.append(run(uniq, analyses, llm, use_judge))

    if not args.generic_only:
        s = pd.read_csv(config.EVAL_DIR / "sample_policy_qa.csv").fillna("")
        rows = [{"set": "policy_specific", "id": t.qa_id,
                 "policy": get(config.SAMPLE_POLICY_DIR / t.policy_file), "question": t.question,
                 "guidance": t.expected_answer, "avoid": "Do not invent terms or give buy/switch advice.",
                 "keywords": t.expected_keywords, "answerable": t.answerable} for t in s.itertuples()]
        frames.append(run(rows, analyses, llm, use_judge))

    df = pd.concat(frames, ignore_index=True)
    out = config.REPORTS_DIR / f"qa_eval_{'offline' if not llm.available else llm.provider}.csv"
    df.to_csv(out, index=False)

    print(f"LLM: {llm.label} | questions evaluated: {len(df)}")
    print(f"Citation rate            : {df.has_citation.mean():.0%}")
    print(f"Unsafe buy/switch advice : {df.unsafe_advice.mean():.0%} (target 0%)")
    if "answerable" in df:
        ps = df[df.set == "policy_specific"]
        ans = ps[ps.answerable == "yes"]
        unans = ps[ps.answerable == "no"]
        print(f"Keyword recall (answerable): {ans.keyword_recall.mean():.0%}")
        print(f"Correct refusal (unanswerable): {unans.said_not_found.mean():.0%}")
        print(f"False 'not found' on answerable: {ans.said_not_found.mean():.0%}")
    jcols = [c for c in df.columns if c.startswith("judge_") and c != "judge_hallucination"]
    if jcols:
        print("LLM-as-judge (1-5):", df[jcols].apply(pd.to_numeric, errors="coerce").mean().round(2).to_dict())
        if "judge_hallucination" in df:
            print(f"Judge-flagged hallucination: {df.judge_hallucination.fillna(False).astype(bool).mean():.0%}")
    print("saved", out)
