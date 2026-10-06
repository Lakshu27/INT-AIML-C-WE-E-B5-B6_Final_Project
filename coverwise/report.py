"""PDF Coverage Insight Report (ReportLab)."""
from __future__ import annotations

import datetime as dt
import io
import re
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

from . import config

BRAND = colors.HexColor("#0F4C81")
LABEL_COLORS = {"Strong Coverage": "#1B7F3B", "Moderate Coverage": "#2F6DB5",
                "Caution Required": "#C77700", "High Risk": "#B3261E"}

_ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=_ss["Title"], textColor=BRAND, fontSize=20, spaceAfter=4),
    "sub": ParagraphStyle("s", parent=_ss["Normal"], alignment=TA_CENTER, textColor=colors.grey),
    "h1": ParagraphStyle("h1", parent=_ss["Heading2"], textColor=BRAND, spaceBefore=10, spaceAfter=4),
    "h2": ParagraphStyle("h2", parent=_ss["Heading3"], spaceBefore=6, spaceAfter=2),
    "body": ParagraphStyle("b", parent=_ss["BodyText"], fontSize=9.5, leading=13),
    "small": ParagraphStyle("sm", parent=_ss["BodyText"], fontSize=8, leading=10),
    "cell": ParagraphStyle("c", parent=_ss["BodyText"], fontSize=8, leading=10),
    "disc": ParagraphStyle("d", parent=_ss["BodyText"], fontSize=8.5, leading=11,
                           backColor=colors.HexColor("#FFF4E5"), borderPadding=6,
                           borderColor=colors.HexColor("#C77700"), borderWidth=0.5),
}


_CHAR_MAP = {"\u20b9": "Rs ", "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2212": "-",
             "\u00a0": " ", "\u202f": " ", "\u2009": " ", "\u200b": "", "\u3010": "[", "\u3011": "]",
             "\u2264": "<=", "\u2265": ">=", "\u2192": "->", "\u2248": "~"}


def _clean(text) -> str:
    """Make text printable with the built-in PDF font (no missing-glyph boxes)."""
    text = str(text)
    for k, v in _CHAR_MAP.items():
        text = text.replace(k, v)
    return text.encode("cp1252", "replace").decode("cp1252").replace("?", "?")


def _rich(text: str) -> str:
    """Escape + convert **bold** markdown; drop unmatched asterisks."""
    safe = escape(_clean(text))
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
    return safe.replace("**", "").replace("`", "")


def _p(text, style="body"):
    return Paragraph(_rich(text), S[style])


def _md(text: str) -> list:
    """Small markdown -> flowables: headings, bullets, bold, and tables (rendered as bullet lines)."""
    out = []
    header = None
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            header = None
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue  # separator row
            if header is None:
                header = cells
                continue
            parts = [f"<b>{_rich(cells[0])}</b>"] + [
                f"{_rich(h)}: {_rich(c)}" for h, c in zip(header[1:], cells[1:]) if c]
            out.append(Paragraph("\u2022 " + " | ".join(parts), S["body"]))
            continue
        header = None
        bare = line.lstrip("-*\u2022 ").strip()
        if bare.startswith("#"):
            out.append(Paragraph(_rich(bare.lstrip("# ")), S["h2"]))
        elif line.startswith(("-", "*", "\u2022")):
            out.append(Paragraph("\u2022 " + _rich(bare), S["body"]))
        else:
            out.append(Paragraph(_rich(line), S["body"]))
    return out


def _table(rows: list[list], widths, header=True, zebra=True):
    data = [[Paragraph(_rich(c), S["cell"]) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C9D3DF")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E3ECF7"))]
    if zebra:
        for i in range(1, len(rows)):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F7F9FC")))
    t.setStyle(TableStyle(style))
    return t


def _ev(ext, field):
    e = (ext.get("evidence") or {}).get(field)
    return f"p.{e['page']}" if isinstance(e, dict) and e.get("page") else "-"


def _val(v):
    if v is None or v == [] or v == "":
        return "Not clearly found"
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, list):
        return "; ".join(str(x) for x in v[:6]) + (" ..." if len(v) > 6 else "")
    return str(v)


CLAUSE_ROWS = [
    ("Insurer", "insurer_name"), ("Policy / plan", "policy_name"), ("UIN", "uin"),
    ("Policy type", "policy_type"), ("Sum insured", "sum_insured"), ("Annual premium", "annual_premium"),
    ("In-patient hospitalisation", "inpatient_hospitalization"),
    ("Pre-hospitalisation days", "pre_hospitalization_days"),
    ("Post-hospitalisation days", "post_hospitalization_days"), ("Day care", "day_care_covered"),
    ("Ambulance", "ambulance_cover"), ("AYUSH", "ayush_covered"), ("Domiciliary", "domiciliary_covered"),
    ("Restoration benefit", "restoration_benefit"), ("No-claim bonus", "no_claim_bonus"),
    ("Maternity", "maternity_covered"), ("Newborn", "newborn_covered"), ("Dependent child", "child_coverage"),
    ("Initial waiting (days)", "initial_waiting_period_days"),
    ("PED waiting (months)", "ped_waiting_period_months"),
    ("Specified disease waiting (months)", "specific_disease_waiting_period_months"),
    ("Co-payment %", "copayment_percentage"), ("Co-payment conditions", "copayment_conditions"),
    ("Room rent limit", "room_rent_limit"), ("ICU limit", "icu_limit"),
    ("Disease sub-limits", "disease_sub_limits"), ("Exclusions (count)", "exclusion_count"),
    ("Cashless", "cashless_available"), ("Reimbursement", "reimbursement_available"),
    ("Claim intimation", "claim_intimation_timeline"), ("Grace period (days)", "grace_period_days"),
    ("Portability described", "portability_mentioned"),
]


def _policy_section(a: dict) -> list:
    ext, sc = a["extraction"], a["score"]
    color = LABEL_COLORS.get(sc["label"], "#333333")
    flow = [Paragraph(escape(a["name"]), S["h1"]),
            Paragraph(f"<b>Coverage Clarity Score:</b> <font color='{color}' size='13'><b>{sc['score']}/100 "
                      f"&#8211; {_rich(sc['label_display'])}</b></font>", S["body"]),
            _p(f"Scoring method: {sc['method']}"
               + (f" | model confidence {sc['confidence']:.0%}" if sc.get("confidence") else "")
               + f" | transparent rule score {sc['rule_score']}/100 ({sc['rule_label']})", "small"),
            Spacer(1, 4), Paragraph("Easy-to-read summary", S["h2"])]
    flow += [_p(l.lstrip("- ").strip()) for l in a["summary"]]
    flow.append(Paragraph("Top reasons behind the score", S["h2"]))
    rows = [["Factor", "Value", "Points"]] + [
        [r["description"], str(r["value"]), f"{r['points']:+}"] for r in sc["reasons"][:8]]
    flow.append(_table(rows, [95 * mm, 35 * mm, 25 * mm]))
    flow.append(Paragraph("Coverage and restriction report", S["h2"]))
    rows = [["Clause", "Extracted value", "Source"]] + [
        [label, _val(ext.get(field))[:220], _ev(ext, field)] for label, field in CLAUSE_ROWS]
    flow.append(_table(rows, [48 * mm, 110 * mm, 17 * mm]))
    flow.append(Paragraph("Risk flags and caution points", S["h2"]))
    rows = [["Severity", "Flag", "Why it matters", "Source"]] + [
        [f["severity"], f["flag"], f["why"], f["source"]] for f in a["flags"]]
    flow.append(_table(rows, [17 * mm, 55 * mm, 88 * mm, 15 * mm]))
    if ext.get("ambiguous_clauses"):
        flow.append(Paragraph("Unclear clauses to verify", S["h2"]))
        flow += [_p("\u2022 " + c[:220], "small") for c in ext["ambiguous_clauses"][:8]]
    return flow


def generate_report(result: dict, profile: dict | None = None, qa_log: list | None = None) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm,
                            title="CoverWise AI - Coverage Insight Report")
    flow = [Paragraph("CoverWise AI", S["title"]),
            _p("Coverage Insight and Risk Intelligence Report", "sub"),
            _p(f"Generated {dt.datetime.now():%d %b %Y, %H:%M}", "sub"), Spacer(1, 8),
            Paragraph("<b>Important:</b> " + escape(config.DISCLAIMER), S["disc"]), Spacer(1, 6)]

    if profile:
        flow.append(Paragraph("User profile used for this analysis", S["h1"]))
        flow.append(_table([["Field", "Value"]] + [[k.replace("_", " ").title(), _val(v)]
                                                   for k, v in profile.items() if v not in (None, "", [])],
                           [60 * mm, 115 * mm]))

    flow += _policy_section(result["current"])
    if result.get("proposed"):
        flow.append(PageBreak())
        flow += _policy_section(result["proposed"])
        cmp_ = result["comparison"]
        flow.append(PageBreak())
        flow.append(Paragraph("Current policy vs proposed policy", S["h1"]))
        rows = [["Aspect", "Current", "Proposed", "Change"]] + [
            [r["Aspect"], r["Current"], r["Proposed"], r["Change"]] for r in cmp_["rows"]]
        flow.append(_table(rows, [62 * mm, 42 * mm, 42 * mm, 29 * mm]))
        flow.append(Spacer(1, 4))
        flow.append(Paragraph("This policy appears stronger in these areas", S["h2"]))
        flow.append(_p(", ".join(cmp_["improved"]) or "None detected"))
        flow.append(Paragraph("This policy needs caution in these areas", S["h2"]))
        flow.append(_p(", ".join(cmp_["worse"]) or "None detected"))
        flow.append(Paragraph("Unclear - must be verified", S["h2"]))
        flow.append(_p(", ".join(cmp_["unclear"]) or "None"))
        flow.append(Paragraph("Premium vs coverage", S["h2"]))
        flow.append(_p(cmp_["premium_note"]))
        flow.append(Paragraph("Switching risks", S["h2"]))
        flow += [_p("\u2022 " + s) for s in cmp_["switching_risks"] + cmp_["profile_cautions"]]
        if cmp_.get("narrative"):
            flow.append(Paragraph("AI explanation of the comparison", S["h2"]))
            flow += _md(cmp_["narrative"])

    questions = (result.get("proposed") or result["current"])["advisor_questions"]
    flow.append(KeepTogether([Paragraph("Questions to ask the insurance advisor / insurer", S["h1"])]
                             + [_p(f"{i}. {q}") for i, q in enumerate(questions, 1)]))
    if qa_log:
        flow.append(Paragraph("Your policy questions (from this session)", S["h1"]))
        for item in qa_log[-8:]:
            flow.append(Paragraph(f"<b>Q:</b> {_rich(item['question'])}", S["body"]))
            flow += _md(item["answer"])
            flow.append(Spacer(1, 3))

    flow.append(Paragraph("Responsible AI note and limitations", S["h1"]))
    for line in [
        config.DISCLAIMER,
        "Values were extracted automatically from the uploaded document(s). Page references are given so "
        "you can check each value in the original wording.",
        "Fields marked 'Not clearly found' were not stated clearly; typical values were assumed only for "
        "scoring and are listed as 'Verify' flags.",
        "The Coverage Clarity Score is trained on a synthetic dataset and describes how clear / restrictive "
        "the wording is - it is not a measure of claim settlement or insurer quality.",
        "Personal details (names, phone numbers, emails, PAN, Aadhaar, policy numbers) are masked before "
        "any text is sent to the language model.",
    ]:
        flow.append(_p("\u2022 " + line))

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(16 * mm, 8 * mm, "CoverWise AI - decision support only, not insurance advice")
        canvas.drawRightString(A4[0] - 16 * mm, 8 * mm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(flow, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()