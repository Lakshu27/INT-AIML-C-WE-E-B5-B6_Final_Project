"""RAG-based Policy Q&A Engine + LLM summary generation (with offline fallbacks)."""
from __future__ import annotations

import re

from . import config
from .analysis import is_advice_question, sanitize
from .llm import LLMClient, LLMError
from .prompts import render, system_prompt
from .rag import PolicyIndex, _expand_query
from .text_clean import split_sentences

NOT_FOUND = "Not clearly found in the uploaded document."
_STOP = set("""the a an is are of in to for and or this that my policy does do what any there be with on
under i me how can will it its as at by from have has which when where why who
benefit benefits cover covered coverage expenses insured sum person policyholder""".split())


# question words that are not "facts" - their absence does not mean the topic is missing
_GENERIC = set("""applicable apply applies capped cap limited limit limits separately process major main always
better best worse low high much many long period available allowed included include get give need required
take taken treatment treatments insurer company verify risks risk switching before after should buy
current proposed plan exactly know""".split())


_CAT_LABEL = {"exclusions": "EXCLUSIONS - not covered", "waiting_period": "WAITING PERIODS",
              "coverage": "COVERAGE / BENEFITS", "claims": "CLAIMS", "renewal": "RENEWAL / PORTABILITY",
              "definitions": "DEFINITIONS", "copay_limits": "CO-PAYMENT / LIMITS", "premium": "PREMIUM / ELIGIBILITY"}


def _context(hits) -> str:
    """Page-tagged context; the section type tells the LLM whether a passage grants or excludes cover."""
    return "\n\n".join(
        f"[S{i}] ({c.citation}) [section type: {_CAT_LABEL.get(c.section_category, c.section_category)}]\n{c.text}"
        for i, (c, _) in enumerate(hits, 1))


def _extractive_answer(question: str, hits) -> str:
    tok = lambda t: {w for w in re.findall(r"[a-z][a-z\-]+", t.lower()) if w not in _STOP}
    orig, expanded = tok(question), tok(_expand_query(question))
    definitional = bool(re.search(r"\bmean|defin", question, re.I))
    scored = []
    for i, (chunk, _) in enumerate(hits, 1):
        for sent in split_sentences(chunk.text.replace("\n", " ")):
            words = tok(sent)
            score = 2 * len(orig & words) + len(expanded & words) - 0.3 * i
            if re.search(r"\d", sent):
                score += 0.5
            if " means " in sent.lower() and not definitional:
                score -= 2
            if len(orig & words) and len(sent) > 30:
                scored.append((score, i, sent.strip()))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return NOT_FOUND
    # every key term of the question (or a synonym) must appear somewhere in the retrieved text
    from .rag import _SYNONYMS
    hit_words = set().union(*(tok(c.text) for c, _ in hits))
    missing = [t for t in orig - _GENERIC if t not in hit_words and not (tok(_SYNONYMS.get(t, "")) & hit_words)
               and len(t) > 2]
    prefix = ""
    if missing:
        prefix = (f"{NOT_FOUND} The document does not mention: {', '.join(sorted(missing))}. "
                  "Closest wording (may not answer your question):\n")
    picked, seen = [], set()
    for _, i, sent in scored:
        if sent[:50] not in seen:
            picked.append(f"- {sent[:350]} [S{i}]")
            seen.add(sent[:50])
        if len(picked) == 3:
            break
    header = prefix or ("Most relevant wording found in the document (offline mode - add an LLM key "
                        "for a plain-English answer):\n")
    return header + "\n".join(picked) + "\n\nVerify: confirm the exact meaning with the insurer/advisor."


def decision_support_answer(analysis: dict) -> str:
    """Used for 'should I buy / switch?' questions: no verdict, only facts + questions."""
    f, ext, sc = analysis["features"], analysis["extraction"], analysis["score"]
    strengths = []
    if f["copayment_percentage"] == 0:
        strengths.append("no general co-payment found")
    if not f["room_rent_limit_present"]:
        strengths.append("no room-rent cap found")
    if f["restoration_benefit_present"]:
        strengths.append("restoration of sum insured available")
    if f["no_claim_bonus_present"]:
        strengths.append("no-claim / cumulative bonus available")
    if f["maternity_coverage_present"]:
        strengths.append("maternity cover available")
    if (ext.get("post_hospitalization_days") or 0) >= 90:
        strengths.append(f"long post-hospitalisation cover ({ext['post_hospitalization_days']} days)")
    cautions = [fl for fl in analysis.get("flags", []) if fl["severity"] in ("High", "Medium")]
    lines = [f"Coverage Clarity Score: **{sc['score']}/100 ({sc['label_display']})**.", "",
             "**Appears stronger in:** " + (", ".join(strengths) or "no clear strengths detected"), "",
             "**Needs caution in:**"]
    lines += [f"- {c['flag']} ({c['source']}) - {c['why']}" for c in cautions[:6]] or ["- none detected"]
    lines += ["", "**Ask these before deciding:**"]
    lines += [f"- {q}" for q in analysis.get("advisor_questions", [])[:5]]
    lines += ["", "Verify: the final decision should be made with the insurer or a licensed advisor."]
    return "\n".join(lines)


def answer_question(index: PolicyIndex, question: str, llm: LLMClient | None = None,
                    profile: dict | None = None, k: int = config.TOP_K, analysis: dict | None = None) -> dict:
    hits = index.search(question, k=k)
    sources = [{"tag": f"S{i}", "citation": c.citation, "page": c.page_start, "text": c.text[:600]}
               for i, (c, _) in enumerate(hits, 1)]
    advice = is_advice_question(question)
    if advice and analysis is not None and (not llm or not llm.available or not hits
                                            or hits[0][1] < config.MIN_RETRIEVAL_SCORE):
        return {"answer": "**CoverWise does not make buy / switch / cancel decisions.** "
                          "Here is a factual summary to discuss with a licensed advisor:\n\n"
                          + decision_support_answer(analysis),
                "sources": sources, "grounded": True, "advice_question": True, "context": _context(hits)}
    if not hits or hits[0][1] < config.MIN_RETRIEVAL_SCORE:
        return {"answer": NOT_FOUND + " Please ask the insurer to point you to the exact clause.",
                "sources": sources, "grounded": False, "advice_question": advice}
    if llm and llm.available:
        try:
            ans = llm.chat(system_prompt(), render("rag_qa", question=question,
                                                   profile=profile or {}, context=_context(hits)))
        except LLMError as exc:
            reason = "daily token limit reached" if "429" in str(exc) else "LLM unavailable"
            if advice and analysis is not None:
                ans = decision_support_answer(analysis)
            else:
                ans = _extractive_answer(question, hits)
            ans += f"\n\n_(Offline answer - {reason}; try again later or switch model.)_"
    else:
        ans = _extractive_answer(question, hits)
    ans = sanitize(ans).replace("\u3010", "[").replace("\u3011", "]")
    if advice:
        ans = ("**CoverWise does not make buy / switch / cancel decisions.** Here is what the document says "
               "so you can discuss it with a licensed advisor:\n\n" + ans)
    if not re.search(r"\[S\d+\]", ans) and NOT_FOUND not in ans:
        ans += "\n\nSources: " + "; ".join(f"[{s['tag']}] {s['citation']}" for s in sources[:3])
    return {"answer": ans, "sources": sources, "grounded": NOT_FOUND not in ans, "advice_question": advice,
            "context": _context(hits)}


# ----------------------------------------------------------------------- summary
def _short(text, n=110):
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "\u2026"


def _yn(v):
    return "Yes" if v is True else ("No" if v is False else "Not clearly stated (verify)")


def template_summary(ext: dict, features: dict, score: dict) -> list[str]:
    lines = [
        f"- This is a {ext.get('policy_type') or 'health'} policy"
        f"{' named ' + ext['policy_name'] if ext.get('policy_name') else ''}"
        f"{' from ' + ext['insurer_name'] if ext.get('insurer_name') else ''}.",
        f"- Sum insured used for analysis: Rs {features['sum_insured']:,}; annual premium: Rs {features['annual_premium']:,}.",
        f"- Hospital stay (in-patient) cover: {_yn(ext.get('inpatient_hospitalization'))}; day-care procedures: "
        f"{_yn(ext.get('day_care_covered'))}.",
        f"- Expenses before/after hospitalisation: {ext.get('pre_hospitalization_days') or '?'} days before and "
        f"{ext.get('post_hospitalization_days') or '?'} days after.",
        f"- Pre-existing diseases are covered only after {ext.get('ped_waiting_period_months') or '? (verify)'} months; "
        f"specified illnesses after {ext.get('specific_disease_waiting_period_months') or '? (verify)'} months.",
        f"- New illnesses (other than accidents) are not covered for the first "
        f"{ext.get('initial_waiting_period_days') or '? (verify)'} days.",
        f"- Co-payment: {features['copayment_percentage']:g}% of each claim is paid by you"
        f"{' (' + _short(ext['copayment_conditions']) + ')' if ext.get('copayment_conditions') else ''}.",
        f"- Room rent limit: {_yn(ext.get('room_rent_limit_present'))}"
        f"{' (' + _short(ext['room_rent_limit']) + ')' if ext.get('room_rent_limit') else ''}.",
        f"- ICU limit: {_yn(ext.get('icu_limit_present'))}.",
        f"- Disease-wise sub-limits found: {features['disease_sub_limit_count']}; exclusions listed: "
        f"about {features['exclusion_count']}.",
        f"- Restoration of sum insured: {_yn(ext.get('restoration_benefit'))}; no-claim bonus: "
        f"{_yn(ext.get('no_claim_bonus'))}.",
        f"- Maternity: {_yn(ext.get('maternity_covered'))}; newborn: {_yn(ext.get('newborn_covered'))}; "
        f"dependent child: {_yn(ext.get('child_coverage'))}.",
        f"- Claims: cashless {_yn(ext.get('cashless_available'))}, reimbursement "
        f"{_yn(ext.get('reimbursement_available'))}; claim process clarity {features['claim_process_clarity_score']}/100.",
        f"- Coverage Clarity Score: {score['score']}/100 ({score['label_display']}).",
        f"- {len(ext.get('ambiguous_clauses') or [])} vague / schedule-dependent clauses need verification.",
    ]
    return lines


def policy_summary(ext: dict, features: dict, score: dict, index: PolicyIndex,
                   llm: LLMClient | None = None) -> list[str]:
    if llm and llm.available:
        hits = index.search("benefits covered waiting period co-payment room rent exclusions claim", k=6)
        try:
            from .analysis import _slim
            text = llm.chat(system_prompt(), render("summary", extraction=_slim(ext), context=_context(hits)))
            text = re.sub(r"\s*[\[\u3010]\s*(extraction\s*)?json\s*[\]\u3011]", "", text, flags=re.I)
            text = text.replace("\u3010", "[").replace("\u3011", "]")
            lines = [l.strip() for l in sanitize(text).splitlines() if l.strip().startswith(("-", "*", "\u2022"))]
            lines = [l.replace("**", "").replace("__", "") for l in lines]  # plain text, no stray markdown
            if 6 <= len(lines) <= 25:
                return ["- " + l.lstrip("-*\u2022 ").strip() for l in lines]
        except LLMError:
            pass
    return template_summary(ext, features, score)