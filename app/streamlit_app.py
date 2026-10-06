"""CoverWise AI - Streamlit dashboard.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coverwise import config  # noqa: E402
from coverwise.llm import LLMClient  # noqa: E402
from coverwise.pipeline import analyze_pair  # noqa: E402
from coverwise.qa import answer_question  # noqa: E402
from coverwise.report import CLAUSE_ROWS, generate_report  # noqa: E402

st.set_page_config(page_title="CoverWise AI", page_icon="\U0001F6E1\uFE0F", layout="wide")

LABEL_COLOR = {"Strong Coverage": "#1B7F3B", "Moderate Coverage": "#2F6DB5",
               "Caution Required": "#C77700", "High Risk": "#B3261E"}
CHANGE_COLOR = {"Improved": "#E7F4EA", "Worse": "#FBE9E7", "Unclear": "#FFF4E0",
                "Higher cost": "#FBE9E7", "Lower cost": "#E7F4EA"}
MAX_MB = 25

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
h1, h2, h3 { font-family: 'Source Serif 4', Georgia, serif !important; letter-spacing: -0.01em; }
.cw-hero { border-left: 6px solid #0F4C81; padding: 0.4rem 0 0.4rem 1rem; margin-bottom: 0.6rem; }
.cw-hero h1 { margin: 0; font-size: 2.3rem; color: #0F4C81; }
.cw-hero p { margin: 0.2rem 0 0; color: #5A6472; }
.cw-stamp { border: 2px solid var(--c); border-radius: 10px; padding: 0.9rem 1.1rem; }
.cw-stamp .num { font-family: 'Source Serif 4', serif; font-size: 2.8rem; font-weight: 700; color: var(--c); line-height: 1; }
.cw-stamp .lbl { font-weight: 600; color: var(--c); text-transform: uppercase; letter-spacing: .06em; font-size: .8rem; }
.cw-stamp .sub { color: #5A6472; font-size: .8rem; margin-top: .35rem; }
.cw-disc { background: #FFF6E8; border: 1px solid #E5B567; border-radius: 8px; padding: .6rem .9rem; font-size: .86rem; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("1. Language model")
    provider = st.selectbox("Provider", ["groq", "gemini", "none (offline rule-based)"],
                            index=["groq", "gemini"].index(config.LLM_PROVIDER)
                            if config.LLM_PROVIDER in ("groq", "gemini") else 2)
    provider = provider.split()[0]
    default_key = config.GROQ_API_KEY if provider == "groq" else config.GEMINI_API_KEY
    default_model = config.GROQ_MODEL if provider == "groq" else config.GEMINI_MODEL
    api_key, model = "", ""
    if provider != "none":
        api_key = st.text_input("API key (kept only in this session)", value=default_key, type="password")
        model = st.text_input("Model", value=default_model)
        if not api_key:
            st.caption("No key -> the app falls back to offline rule-based mode.")

    st.header("2. Your profile")
    age = st.number_input("Age of eldest insured person", 0, 100, 35)
    family = st.multiselect("Family members to cover", ["Self", "Spouse", "Child", "Parents"],
                            default=["Self", "Spouse"])
    has_ped = st.checkbox("Someone has a pre-existing condition")
    planning_child = st.checkbox("Planning a child (maternity)")
    adding_child = st.checkbox("Want to add a child / newborn")
    concern = st.text_area("Main concern (optional)", placeholder="e.g. switching from employer cover")

    st.header("3. Premium / sum insured")
    st.caption("Fill only if the document does not show your actual values (0 = auto-detect).")
    c1, c2 = st.columns(2)
    cur_prem = c1.number_input("Current premium Rs", 0, step=500)
    cur_si = c2.number_input("Current SI Rs", 0, step=100000, help="e.g. 500000 = Rs 5 lakh")
    prop_prem = c1.number_input("Proposed premium Rs", 0, step=500)
    prop_si = c2.number_input("Proposed SI Rs", 0, step=100000, help="e.g. 1000000 = Rs 10 lakh")
    for label, v in (("Current", cur_si), ("Proposed", prop_si)):
        if v:
            st.caption(f"{label} SI = Rs {v / 100000:g} lakh")

profile = {"age": int(age), "family_members": ", ".join(family), "has_ped": has_ped,
           "planning_child": planning_child, "adding_child": adding_child, "concern": concern or None}

# ------------------------------------------------------------------ header + upload
st.markdown("<div class='cw-hero'><h1>CoverWise AI</h1>"
            "<p>Read the fine print before the hospital does. Upload a health policy (and optionally a "
            "proposed one) to decode clauses, compare coverage and spot claim-time risks.</p></div>",
            unsafe_allow_html=True)
st.markdown(f"<div class='cw-disc'>\u26A0\uFE0F {config.DISCLAIMER}</div>", unsafe_allow_html=True)
st.write("")

u1, u2 = st.columns(2)
cur_file = u1.file_uploader("Current policy PDF (wording / brochure / CIS)", type=["pdf"])
prop_file = u2.file_uploader("Proposed policy PDF (optional)", type=["pdf"])
use_samples = st.checkbox("Use the bundled fictional sample policies instead", value=not cur_file)
st.caption("Privacy: names, phone numbers, emails, PAN, Aadhaar and policy numbers are masked before any "
           "text is sent to the LLM. Do not upload real customer documents to a public demo.")

if st.button("Analyze policy", type="primary", width="stretch"):
    if use_samples:
        cur_src = config.SAMPLE_POLICY_DIR / "current_policy_nilgiri_arogya_basic.pdf"
        prop_src = config.SAMPLE_POLICY_DIR / "proposed_policy_nilgiri_arogya_plus.pdf"
    else:
        cur_src, prop_src = cur_file, prop_file
    if cur_src is None:
        st.error("Please upload the current policy PDF (or tick 'Use the bundled sample policies').")
        st.stop()
    for f in (cur_file, prop_file):
        if f is not None and not use_samples and f.size > MAX_MB * 1024 * 1024:
            st.error(f"{f.name} is larger than {MAX_MB} MB.")
            st.stop()
    llm = LLMClient(provider, api_key or None, model or None)
    bar = st.progress(0.0, text="Starting...")
    n_docs = 2 if prop_src is not None else 1
    state = {"doc": 0, "last": 0.0}

    def progress(frac, msg):
        if frac < state["last"]:
            state["doc"] += 1
        state["last"] = frac
        bar.progress(min((state["doc"] + frac) / n_docs, 1.0), text=msg)

    try:
        result = analyze_pair(cur_src, prop_src, llm, profile,
                              {"annual_premium": cur_prem, "sum_insured": cur_si},
                              {"annual_premium": prop_prem, "sum_insured": prop_si}, progress)
    except Exception as exc:  # show a friendly error instead of a stack trace
        bar.empty()
        st.error(f"Analysis failed: {exc}")
        st.stop()
    bar.empty()
    st.session_state.update(result=result, llm=llm, qa_log=[], profile=profile)
    st.success(f"Done with {llm.label}. Retrieval: {result['current']['index'].mode}.")

result = st.session_state.get("result")
if not result:
    st.info("Upload a policy (or tick the sample option) and click **Analyze policy**.")
    st.stop()

llm: LLMClient = st.session_state["llm"]
policies = [result["current"]] + ([result["proposed"]] if result["proposed"] else [])


def stamp(a):
    sc = a["score"]
    col = LABEL_COLOR.get(sc["label"], "#333")
    conf = f" \u00b7 model confidence {sc['confidence']:.0%}" if sc.get("confidence") else ""
    note = "" if sc.get("agreement", True) else " \u00b7 borderline: rule score band differs"
    st.markdown(f"<div class='cw-stamp' style='--c:{col}'><div class='lbl'>{a['name']}</div>"
                f"<div class='num'>{sc['score']}<span style='font-size:1.1rem'>/100</span></div>"
                f"<div class='lbl'>{sc['label_display']}</div>"
                f"<div class='sub'>{sc['method']}{conf}{note}<br>{a['pages']} pages \u00b7 "
                f"{a['features']['risk_flags_count']} risk flags \u00b7 {a['processing_seconds']}s</div></div>",
                unsafe_allow_html=True)


tabs = st.tabs(["Summary", "Clauses", "Score & risk flags", "Comparison", "Ask the policy",
                "Advisor questions", "Report"])

# ------------------------------------------------------------------ Summary
with tabs[0]:
    cols = st.columns(len(policies))
    for col, a in zip(cols, policies):
        with col:
            stamp(a)
            st.subheader("Plain-English summary")
            st.markdown("\n".join(a["summary"]))

# ------------------------------------------------------------------ Clauses
with tabs[1]:
    pick = st.radio("Policy", [a["name"] for a in policies], horizontal=True, key="cl_pick")
    a = next(p for p in policies if p["name"] == pick)
    ext = a["extraction"]

    def fmt(v):
        if v is None or v == [] or v == "":
            return "Not clearly found"
        if isinstance(v, bool):
            return "Yes" if v else "No"
        if isinstance(v, list):
            return "; ".join(map(str, v))
        return str(v)

    rows = []
    for label, field in CLAUSE_ROWS:
        ev = (ext.get("evidence") or {}).get(field) or {}
        rows.append({"Clause": label, "Extracted value": fmt(ext.get(field)),
                     "Page": str(ev.get("page", "")) if isinstance(ev, dict) else "",
                     "Evidence": ev.get("quote", "") if isinstance(ev, dict) else ""})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=620)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Exclusions found** (" + str(ext.get("exclusion_count") or "?") + ")")
        for e in (ext.get("permanent_exclusions") or [])[:25]:
            st.markdown(f"- {e}")
    with c2:
        st.markdown("**Unclear / schedule-dependent clauses**")
        for c in ext.get("ambiguous_clauses") or ["None detected"]:
            st.markdown(f"- {c}")
        if ext.get("validation_notes"):
            st.markdown("**Validation notes (LLM vs rule cross-check)**")
            for n in ext["validation_notes"]:
                st.warning(n)
    st.caption(f"Extraction method: {ext.get('extraction_method')}. Check every value against the page shown.")

# ------------------------------------------------------------------ Score
with tabs[2]:
    pick = st.radio("Policy", [a["name"] for a in policies], horizontal=True, key="sc_pick")
    a = next(p for p in policies if p["name"] == pick)
    sc = a["score"]
    c1, c2 = st.columns([1, 2])
    with c1:
        stamp(a)
        if sc.get("probabilities"):
            st.markdown("**Class probabilities**")
            st.bar_chart(pd.Series(sc["probabilities"]).reindex(config.RISK_LABELS).fillna(0))
    with c2:
        st.markdown(f"**Top reasons behind the score** (transparent rule score: {sc['rule_score']}/100, "
                    f"{sc['rule_label']})")
        st.dataframe(pd.DataFrame([{"Factor": r["description"], "Value": str(r["value"]), "Points": r["points"]}
                                   for r in sc["reasons"]]), hide_index=True, width="stretch")
        st.markdown("**Features sent to the model**")
        st.dataframe(pd.DataFrame([a["features"]]).T.rename(columns={0: "value"}).astype(str), width="stretch")
    st.subheader("Risk flags and caution points")
    sev_icon = {"High": "\U0001F534", "Medium": "\U0001F7E0", "Verify": "\U0001F7E1"}
    for f in a["flags"]:
        st.markdown(f"{sev_icon.get(f['severity'], '')} **{f['flag']}** \u2014 {f['why']} "
                    f"<span style='color:#5A6472'>({f['source']})</span>", unsafe_allow_html=True)

# ------------------------------------------------------------------ Comparison
with tabs[3]:
    cmp_ = result.get("comparison")
    if not cmp_:
        st.info("Upload a proposed policy to compare.")
    else:
        df = pd.DataFrame(cmp_["rows"])
        st.dataframe(df.style.apply(lambda r: [f"background-color: {CHANGE_COLOR.get(r['Change'], '')}"] * len(r),
                                    axis=1), width="stretch", hide_index=True, height=760)
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Appears stronger in**\n\n" + ("\n".join(f"- {x}" for x in cmp_["improved"]) or "-"))
        c2.markdown("**Needs caution in**\n\n" + ("\n".join(f"- {x}" for x in cmp_["worse"]) or "- none detected"))
        c3.markdown("**Unclear - verify**\n\n" + ("\n".join(f"- {x}" for x in cmp_["unclear"]) or "- none"))
        st.markdown(f"**Premium vs coverage:** {cmp_['premium_note']}")
        st.markdown("**Switching risks**\n\n" + "\n".join(f"- {s}" for s in cmp_["switching_risks"]
                                                         + cmp_["profile_cautions"]))
        if cmp_.get("narrative"):
            with st.expander("AI explanation of the comparison", expanded=True):
                st.markdown(cmp_["narrative"])

# ------------------------------------------------------------------ Q&A
with tabs[4]:
    pick = st.radio("Ask about", [a["name"] for a in policies], horizontal=True, key="qa_pick")
    a = next(p for p in policies if p["name"] == pick)
    examples = ["Is co-payment applicable in this policy?", "What is the waiting period for pre-existing diseases?",
                "Is room rent capped?", "Does the policy cover newborn baby or dependent child?",
                "What are the major exclusions?", "What should I verify before switching from my current policy?"]
    ex = st.selectbox("Example questions", examples)
    ask_ex = st.button("Ask this example")
    q = st.chat_input("Ask a question about this policy")
    question = q or (ex if ask_ex else None)
    for item in st.session_state["qa_log"]:
        with st.chat_message("user"):
            st.write(f"[{item['policy']}] {item['question']}")
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
    if question:
        with st.chat_message("user"):
            st.write(f"[{a['name']}] {question}")
        with st.chat_message("assistant"):
            with st.spinner("Searching the policy..."):
                r = answer_question(a["index"], question, llm, st.session_state["profile"], analysis=a)
            st.markdown(r["answer"])
            with st.expander("Sources used"):
                for s in r["sources"]:
                    st.markdown(f"**[{s['tag']}] {s['citation']}**")
                    st.caption(s["text"])
        st.session_state["qa_log"].append({"policy": a["name"], "question": question, "answer": r["answer"]})

# ------------------------------------------------------------------ Advisor
with tabs[5]:
    target = result["proposed"] or result["current"]
    st.subheader("Questions to ask the insurance advisor / insurer")
    for i, q in enumerate(target["advisor_questions"], 1):
        st.markdown(f"{i}. {q}")
    st.caption("Appears stronger / needs caution / unclear points are listed in the Comparison and Score tabs. "
               "CoverWise never tells you to buy, switch or cancel.")

# ------------------------------------------------------------------ Report
with tabs[6]:
    st.write("The report contains the summary, clause table with page references, score and reasons, "
             "risk flags, comparison, advisor questions, your Q&A and the responsible-AI note.")
    pdf = generate_report(result, st.session_state["profile"], st.session_state["qa_log"])
    st.download_button("Download PDF report", pdf, file_name="coverwise_report.pdf",
                       mime="application/pdf", type="primary")