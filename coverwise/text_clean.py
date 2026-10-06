"""Step 2 - Text preprocessing and structuring.

* removes repeated headers/footers, page numbers and formatting noise
* masks personal data (privacy & anonymization requirement)
* splits the document into sections (coverage, exclusions, waiting period,
  claims, renewal, definitions, ...) while keeping page numbers per line
"""
from __future__ import annotations

import re
from collections import Counter

# ----------------------------------------------------------------- privacy
_PII_PATTERNS = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[EMAIL]"),
    (re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"), "[PHONE]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN]"),
    (re.compile(r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)"), "[AADHAAR]"),
    (re.compile(r"(?i)(policy\s*(?:no|number)\.?\s*[:\-]?\s*)([A-Z0-9/\-]{6,})"), r"\1[POLICY_NO]"),
    (re.compile(r"(?i)((?:insured|proposer|policyholder)(?:'s)?\s*name\s*[:\-]\s*)([A-Za-z .]{3,40})"),
     r"\1[NAME]"),
    (re.compile(r"(?i)(date of birth|dob)\s*[:\-]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"), r"\1: [DOB]"),
]


def anonymize(text: str) -> str:
    """Mask common Indian PII (email, phone, PAN, Aadhaar, policy no., names, DOB)."""
    for pattern, repl in _PII_PATTERNS:
        text = pattern.sub(repl, text)
    return text


# ----------------------------------------------------------------- cleaning
_PAGE_NO = re.compile(r"^\s*(page\s*)?\d{1,3}(\s*(of|/)\s*\d{1,3})?\s*$", re.I)
_NOISE = re.compile(r"[\u2022\u25cf\u25aa\u2013\u2014\uf0b7\uf0a7]")


def _normalise_line(line: str) -> str:
    line = _NOISE.sub(lambda m: "-" if m.group() in "\u2013\u2014" else "\u2022", line)
    line = line.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    line = re.sub(r"[ \t\u00a0]+", " ", line)
    return line.strip()


def _repeated_lines(pages: list[dict]) -> set[str]:
    """Lines that appear at the top/bottom of >= 50% of pages are headers/footers."""
    if len(pages) < 3:
        return set()
    counter: Counter = Counter()
    for p in pages:
        lines = [_normalise_line(l) for l in p["text"].splitlines() if l.strip()]
        # headers/footers: page edges, plus any short line (content-order extraction can put them mid-page)
        edge = set(lines[:3] + lines[-3:]) | {l for l in lines if len(l) < 140}
        counter.update(re.sub(r"\d+", "#", l) for l in edge)
    threshold = max(2, int(0.5 * len(pages)))
    return {l for l, c in counter.items() if c >= threshold}


def clean_pages(pages: list[dict], mask_pii: bool = True) -> list[dict]:
    repeated = _repeated_lines(pages)
    cleaned = []
    for p in pages:
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", p["text"])  # de-hyphenate line breaks
        out = []
        for raw in text.splitlines():
            line = _normalise_line(raw)
            if not line or _PAGE_NO.match(line):
                continue
            if re.sub(r"\d+", "#", line) in repeated:
                continue
            out.append(line)
        body = "\n".join(out)
        if mask_pii:
            body = anonymize(body)
        cleaned.append({"page": p["page"], "text": body, "ocr": p.get("ocr", False)})
    return cleaned


# ----------------------------------------------------------------- sections
SECTION_KEYWORDS = [
    ("waiting_period", ["waiting period", "pre-existing", "pre existing", "excl01", "excl02", "excl03"]),
    ("exclusions", ["exclusion", "not covered", "what is not covered", "non-payable", "non payable"]),
    ("claims", ["claim", "cashless", "reimbursement", "tpa"]),
    ("renewal", ["renewal", "portability", "migration", "grace period", "cancellation",
                 "free look", "moratorium"]),
    ("definitions", ["definition", "meaning of words", "interpretation"]),
    ("copay_limits", ["co-pay", "copay", "co-payment", "sub-limit", "sub limit", "room rent",
                      "limits of coverage", "deductible"]),
    ("coverage", ["coverage", "benefit", "cover", "what is covered", "scope", "hospitali",
                  "maternity", "newborn", "day care", "ambulance", "ayush", "domiciliary",
                  "restoration", "recharge", "bonus"]),
    ("premium", ["premium", "sum insured", "schedule", "eligibility", "entry age"]),
    ("general", ["general terms", "terms and conditions", "conditions", "grievance", "ombudsman"]),
]

_HEADING = re.compile(
    r"^(section\s+[a-z0-9]+\b|part\s+[a-z0-9]+\b|[a-z]?\d{1,2}(\.\d{1,2}){0,3}[.)]?\s*[A-Za-z(]|"
    r"[a-z]\.\d{1,2}(\.\d{1,2})?\s+[A-Z]|code\s*[-:]?\s*excl\s?\d+)",
    re.I,
)


def categorise(heading: str) -> str | None:
    h = heading.lower()
    code = re.search(r"excl\s?-?\s?(\d{1,2})", h)
    if code:  # IRDAI standard codes: Excl01-03 are waiting periods, the rest exclusions
        return "waiting_period" if int(code.group(1)) <= 3 else "exclusions"
    for cat, keys in SECTION_KEYWORDS:
        if any(k in h for k in keys):
            return cat
    return None


def _is_upper(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    return len(letters) >= 5 and sum(c.isupper() for c in letters) / len(letters) > 0.8


def is_heading(line: str) -> bool:
    if len(line) > 95 or len(line) < 3 or line.rstrip().endswith((",", ";", "/", "-", "&")):
        return False
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 3:
        return False
    words = line.split()
    if _is_upper(line) and len(words) <= 12 and not line.endswith("."):
        return True
    if _HEADING.match(line) and len(words) <= 12 and not line.rstrip().endswith("."):
        return True
    if re.search(r"code\s*[-:]?\s*excl\s?\d{1,2}", line, re.I) and len(words) <= 14:
        return True  # e.g. "o. Maternity: Code - Excl18:"
    return False


_NUM = re.compile(r"^(?:(?:section|part)\s+([a-z]|\d{1,2})\b|([a-z])\.(\d{1,2})|(\d{1,2})((?:\.\d{1,2})*)\.?)", re.I)
STRONG_PARENTS = {"exclusions", "waiting_period", "claims", "renewal", "definitions"}


def _section_number(line: str):
    """('n', 7, depth) for '7.15 ...', ('a', 'd', depth) for 'SECTION D' / 'D.1'."""
    m = _NUM.match(line.strip())
    if not m:
        return None
    if m.group(1):
        v = m.group(1).lower()
        return ("n", int(v), 1) if v.isdigit() else ("a", v, 1)
    if m.group(2):
        return ("a", m.group(2).lower(), 2)
    if m.group(4):
        return ("n", int(m.group(4)), 1 + (m.group(5) or "").count("."))
    return None


def split_sections(pages: list[dict]) -> list[dict]:
    """Return blocks: {section_title, section_category, lines: [(page, text)]}.

    Numbered list items inside a section ("1. Hernia", "iv. day care") are not
    treated as new sections: a numeric heading must not go *backwards* relative
    to the current top-level section number, and sub-sections (7.15 under 7)
    inherit a strong parent category such as exclusions / waiting period.
    """
    blocks: list[dict] = []
    current = {"section_title": "Preamble", "section_category": "general", "lines": []}
    cur_top: dict[str, object] = {}
    parent_cat: dict[tuple, str] = {}
    for p in pages:
        for line in p["text"].splitlines():
            heading = is_heading(line)
            num = _section_number(line) if heading else None
            if num and num[0] == "n" and "n" in cur_top and not _is_upper(line):
                top, depth = num[1], num[2]
                # sections are sequential: "9. Cataract" right after section 6 is a list item
                if top < cur_top["n"] or (depth == 1 and top > cur_top["n"] + 2):
                    heading = False
                elif depth == 1 and categorise(line) is None:
                    heading = False  # "9. Cataract and age related eye ailments" inside a list
            if heading and re.match(r"^\s*(section|part)\s+[a-z0-9]+\b", line, re.I):
                cur_top.pop("n", None)  # numbering restarts inside each lettered section
            if heading:
                cat = categorise(line)
                if num:
                    kind, top, depth = num
                    cur_top[kind] = top
                    pc = parent_cat.get((kind, top))
                    if depth == 1:
                        parent_cat[(kind, top)] = cat or current["section_category"]
                    elif pc in STRONG_PARENTS and not re.search(r"excl\s?-?\s?\d", line, re.I):
                        cat = pc
                if current["lines"]:
                    blocks.append(current)
                current = {
                    "section_title": line[:90],
                    "section_category": cat or current["section_category"],
                    "lines": [],
                }
            current["lines"].append((p["page"], line))
    if current["lines"]:
        blocks.append(current)
    return blocks


def full_text(pages: list[dict]) -> str:
    return "\n".join(f"[[PAGE {p['page']}]]\n{p['text']}" for p in pages)


_ABBREV = re.compile(r"(?:\b(?:rs|no|nos|e\.g|i\.e|viz|etc|sr|dr|mr|mrs|ms|st|vs|approx|max|min|cl|sec|art)"
                     r"|\b[a-z]|\d)\.$", re.I)


def split_sentences(text: str) -> list[str]:
    """Sentence splitter that does not break on 'Rs.', 'No.', 'e.g.', clause numbers like '2.1'."""
    parts = re.split(r"(?<=[.;])\s+(?=[A-Z(\d\"'\u201c])", text)
    out: list[str] = []
    for part in parts:
        if out and _ABBREV.search(out[-1]):
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return [p.strip() for p in out if p.strip()]