"""Restriction & Risk Analyzer, Policy Comparison Analyzer, advisor questions
and responsible-AI guardrails."""
from __future__ import annotations

import re

from .features import model_flags
from .llm import LLMClient, LLMError
from .prompts import render, system_prompt

# ======================================================================= guardrails
_ADVICE_Q = re.compile(r"\b(should i|shall i|do i|must i|is it (good|better|worth)|which (one|policy) "
                       r"(is|should)|recommend|buy|switch|port|cancel|worth it)\b", re.I)
_UNSAFE = [
    (re.compile(r"\byou should (definitely )?(buy|purchase|switch|cancel|port|take)\b", re.I),
     "you may want to discuss with an advisor whether to"),
    (re.compile(r"\b(i|we) recommend (buying|switching|cancelling|purchasing)\b", re.I),
     "an advisor can help you evaluate"),
    (re.compile(r"\b(do not|don't) (buy|switch|cancel)\b", re.I), "carefully evaluate before you"),
    (re.compile(r"\bthis is the best policy\b", re.I), "this policy appears stronger in some areas"),
]


def is_advice_question(q: str) -> bool:
    return bool(_ADVICE_Q.search(q or ""))


def sanitize(text: str) -> str:
    for pattern, repl in _UNSAFE:
        text = pattern.sub(repl, text)
    return text


# ======================================================================= risk flags
def generate_risk_flags(ext: dict, features: dict, unclear: list[str], profile: dict | None = None) -> list[dict]:
    """Model flags (counted in risk_flags_count) + profile-specific caution points."""
    profile = profile or {}
    flags = []
    for f in model_flags(features):
        flags.append({"severity": "High" if f["feature"] in (
            "pre_existing_disease_waiting_period_months", "copayment_percentage", "room_rent_limit_present")
            else "Medium", "flag": f["flag"], "why": f["why"],
            "source": _src(ext, _FEATURE_TO_FIELD.get(f["feature"]))})
    age = profile.get("age") or 0
    if age >= 60 and features["copayment_percentage"] > 0:
        flags.append({"severity": "High", "flag": f"Co-payment applies and policyholder age is {age}",
                      "why": "Senior-age claims are usually larger, so a % co-pay costs more.",
                      "source": _src(ext, "copayment_percentage")})
    if profile.get("has_ped") and features["pre_existing_disease_waiting_period_months"] > 0:
        flags.append({"severity": "High", "flag": "You reported a pre-existing condition",
                      "why": f"It may not be covered for "
                             f"{features['pre_existing_disease_waiting_period_months']} months.",
                      "source": _src(ext, "ped_waiting_period_months")})
    if profile.get("planning_child") and not ext.get("maternity_covered"):
        flags.append({"severity": "High", "flag": "Maternity cover not found but a child is planned",
                      "why": "Delivery and newborn expenses may not be covered.",
                      "source": _src(ext, "maternity_covered")})
    if profile.get("adding_child") and not ext.get("child_coverage"):
        flags.append({"severity": "Medium", "flag": "Child coverage not clearly found",
                      "why": "Confirm whether / at what age a dependent child can be added.",
                      "source": _src(ext, "child_coverage")})
    for u in unclear:
        flags.append({"severity": "Verify", "flag": f"{u} not clearly found in the document",
                      "why": "A typical value was assumed for scoring - confirm the real value.", "source": "-"})
    for note in ext.get("validation_notes", [])[:5]:
        flags.append({"severity": "Verify", "flag": "Extraction cross-check", "why": note, "source": "-"})
    return flags


_FEATURE_TO_FIELD = {
    "pre_existing_disease_waiting_period_months": "ped_waiting_period_months",
    "copayment_percentage": "copayment_percentage", "room_rent_limit_present": "room_rent_limit_present",
    "icu_limit_present": "icu_limit_present", "disease_sub_limit_count": "disease_sub_limits",
    "restoration_benefit_present": "restoration_benefit",
    "waiting_period_months": "specific_disease_waiting_period_months",
}


def _src(ext: dict, field: str | None) -> str:
    e = (ext.get("evidence") or {}).get(field or "", {})
    return f"p.{e.get('page')}" if isinstance(e, dict) and e.get("page") else "-"


# ======================================================================= comparison
ASPECTS = [  # (label, extraction field or feature key, kind, better)
    ("Sum insured (Rs)", "f:sum_insured", "num", "higher"),
    ("Annual premium (Rs)", "f:annual_premium", "num", "lower"),
    ("Pre-existing disease waiting (months)", "ped_waiting_period_months", "num", "lower"),
    ("Specified disease waiting (months)", "specific_disease_waiting_period_months", "num", "lower"),
    ("Initial waiting period (days)", "initial_waiting_period_days", "num", "lower"),
    ("Co-payment (%)", "copayment_percentage", "num", "lower"),
    ("Room rent capped", "room_rent_limit_present", "bool", False),
    ("ICU charges capped", "icu_limit_present", "bool", False),
    ("Disease-wise sub-limits (count)", "f:disease_sub_limit_count", "num", "lower"),
    ("Exclusions (count)", "exclusion_count", "num", "lower"),
    ("Restoration / recharge benefit", "restoration_benefit", "bool", True),
    ("No-claim / cumulative bonus", "no_claim_bonus", "bool", True),
    ("Pre-hospitalisation (days)", "pre_hospitalization_days", "num", "higher"),
    ("Post-hospitalisation (days)", "post_hospitalization_days", "num", "higher"),
    ("Day care procedures", "day_care_covered", "bool", True),
    ("AYUSH treatment", "ayush_covered", "bool", True),
    ("Maternity cover", "maternity_covered", "bool", True),
    ("Newborn cover", "newborn_covered", "bool", True),
    ("Dependent child cover", "child_coverage", "bool", True),
    ("Claim process clarity (0-100)", "f:claim_process_clarity_score", "num", "higher"),
]


def _get(analysis: dict, key: str):
    if key in ("f:sum_insured", "f:annual_premium"):  # show the real value, not the model-clipped one
        src = key[2:]
        return analysis["overrides"].get(src) or analysis["extraction"].get(src) or analysis["features"].get(src)
    if key.startswith("f:"):
        return analysis["features"].get(key[2:])
    return analysis["extraction"].get(key)


def _fmt(v):
    if v is None:
        return "Not found"
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return f"{v:,}" if isinstance(v, int) and v >= 10_000 else str(v)


def compare_policies(current: dict, proposed: dict, profile: dict | None = None,
                     llm: LLMClient | None = None) -> dict:
    rows = []
    for label, key, kind, better in ASPECTS:
        a, b = _get(current, key), _get(proposed, key)
        unclear_side = (key.startswith("f:") is False) and (a is None or b is None)
        if key in ("f:sum_insured", "f:annual_premium"):  # imputed if missing
            src = key[2:]
            unclear_side = not ((current["extraction"].get(src) or current["overrides"].get(src)) and
                                (proposed["extraction"].get(src) or proposed["overrides"].get(src)))
        if unclear_side:
            verdict = "Unclear"
        elif a == b:
            verdict = "Same"
        elif kind == "bool":
            verdict = "Improved" if b == better else "Worse"
        else:
            verdict = "Improved" if (b > a) == (better == "higher") else "Worse"
        if label.startswith("Annual premium") and verdict != "Unclear" and a != b:
            verdict = "Higher cost" if b > a else "Lower cost"
        if key in ("f:sum_insured", "f:annual_premium"):  # never show the imputed placeholder as a real value
            src = key[2:]
            if not (current["extraction"].get(src) or current["overrides"].get(src)):
                a = None
            if not (proposed["extraction"].get(src) or proposed["overrides"].get(src)):
                b = None
        fa, fb = _fmt(a), _fmt(b)
        if key in ("f:sum_insured", "f:annual_premium"):
            fa = fa.replace("Not found", "Not provided")
            fb = fb.replace("Not found", "Not provided")
        rows.append({"Aspect": label, "Current": fa, "Proposed": fb, "Change": verdict})

    rows.append({"Aspect": "Coverage Clarity Score", "Current": str(current["score"]["score"]),
                 "Proposed": str(proposed["score"]["score"]),
                 "Change": "Improved" if proposed["score"]["score"] > current["score"]["score"]
                 else ("Same" if proposed["score"]["score"] == current["score"]["score"] else "Worse")})

    improved = [r["Aspect"] for r in rows if r["Change"] == "Improved"]
    worse = [r["Aspect"] for r in rows if r["Change"] == "Worse"]
    unclear = [r["Aspect"] for r in rows if r["Change"] == "Unclear"]

    # premium vs coverage
    pc, pp = _get(current, "f:annual_premium"), _get(proposed, "f:annual_premium")
    prem_change = (pp - pc) / pc * 100 if pc else 0
    prem_known = all((x["extraction"].get("annual_premium") or x["overrides"].get("annual_premium"))
                     for x in (current, proposed))
    if prem_known:
        premium_note = (f"Premium changes by {prem_change:+.0f}% while {len(improved)} aspect(s) improve and "
                        f"{len(worse)} become weaker. Check whether the improved aspects matter for your family "
                        f"(e.g. no room-rent cap matters most in metro hospitals).")
    else:
        prem_change = None
        premium_note = (f"Premiums were not found in the documents - enter both premiums in the sidebar to "
                        f"compare cost against the {len(improved)} improved and {len(worse)} weaker aspect(s).")

    switching = [
        "Waiting periods (pre-existing, specified diseases, maternity) normally restart with a new policy. "
        "Under IRDAI portability rules you can ask for credit of the waiting period already served, "
        "usually only up to the previous sum insured - get this in writing before switching.",
        "Any accumulated no-claim / cumulative bonus on the current policy may not fully carry over.",
        "Do not let the current policy lapse until the new one is issued and the portability "
        "request is accepted (keep a gap-free continuity).",
    ]
    if _get(proposed, "ped_waiting_period_months") and (_get(current, "ped_waiting_period_months") or 0) \
            < (_get(proposed, "ped_waiting_period_months") or 0):
        switching.insert(0, "The proposed policy has a LONGER pre-existing disease waiting period.")
    if (_get(proposed, "f:sum_insured") or 0) > (_get(current, "f:sum_insured") or 0):
        switching.append("For the increased part of the sum insured, waiting periods usually apply afresh.")

    profile = profile or {}
    caution = []
    if profile.get("age", 0) >= 55 and _get(proposed, "copayment_percentage"):
        caution.append("Proposed policy has co-payment and policyholder is 55+.")
    if profile.get("planning_child") and not _get(proposed, "maternity_covered"):
        caution.append("Child is planned but maternity cover is not found in the proposed policy.")
    if profile.get("has_ped"):
        caution.append("You reported a pre-existing condition - confirm the exact PED waiting credit on switching.")

    narrative = None
    if llm and llm.available:
        try:
            narrative = sanitize(llm.chat(system_prompt(), render(
                "comparison", profile=profile, table=rows,
                current=_slim(current["extraction"]), proposed=_slim(proposed["extraction"])), max_tokens=4000))
        except LLMError:
            narrative = None
    return {"rows": rows, "improved": improved, "worse": worse, "unclear": unclear,
            "premium_change_pct": round(prem_change, 1) if prem_change is not None else None, "premium_note": premium_note,
            "switching_risks": switching, "profile_cautions": caution, "narrative": narrative}


def _slim(ext: dict) -> dict:
    return {k: v for k, v in ext.items() if k not in ("evidence", "validation_notes") and v not in (None, [], "")}


# ======================================================================= advisor questions
def advisor_questions(analysis: dict, comparison: dict | None = None, profile: dict | None = None,
                      llm: LLMClient | None = None) -> list[str]:
    ext, f = analysis["extraction"], analysis["features"]
    qs = []
    if f["copayment_percentage"] > 0:
        qs.append(f"The policy mentions {f['copayment_percentage']:g}% co-payment - does it apply to all claims "
                  f"or only certain ages / cities / non-network hospitals?")
    if ext.get("room_rent_limit_present"):
        qs.append("What exactly is the room rent limit, and will 'proportionate deduction' reduce other bills "
                  "if I choose a higher room category?")
    if ext.get("icu_limit_present"):
        qs.append("What is the ICU per-day limit and what happens if the ICU bill is above it?")
    qs.append(f"Please confirm the pre-existing disease waiting period "
              f"({ext.get('ped_waiting_period_months') or 'not found'} months) and which of my conditions it covers.")
    if ext.get("disease_sub_limits"):
        qs.append("Which treatments have sub-limits (e.g. cataract, joint replacement) and what are the amounts?")
    if not ext.get("restoration_benefit"):
        qs.append("Is there any restoration / recharge of the sum insured if it is exhausted during the year?")
    qs.append("What is the list of non-payable items / consumables, and is there any add-on to cover them?")
    qs.append("What are the claim intimation timelines and documents needed for cashless and reimbursement?")
    if (profile or {}).get("planning_child") or (profile or {}).get("adding_child"):
        qs.append("From what age can a newborn / child be added, and is maternity covered with what waiting period?")
    if comparison:
        qs.append("If I port to the new policy, how much waiting-period credit will I get, and is it in writing?")
        qs.append("Will my accumulated no-claim / cumulative bonus carry over?")
    for clause in (ext.get("ambiguous_clauses") or [])[:2]:
        qs.append(f"Please explain this clause in simple terms: {clause[:140]}")

    if llm and llm.available:
        try:
            out = llm.chat_json(system_prompt(), render(
                "advisor_questions", profile=profile or {},
                flags=[x["flag"] for x in analysis.get("flags", [])][:12],
                unclear=(ext.get("ambiguous_clauses") or [])[:6],
                comparison=(comparison or {}).get("rows", "n/a")))
            llm_qs = [sanitize(q) for q in out.get("questions", []) if isinstance(q, str)]
            if len(llm_qs) >= 6:
                return llm_qs[:12]
        except LLMError:
            pass
    return qs[:12]