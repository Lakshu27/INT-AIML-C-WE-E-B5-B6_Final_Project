"""Policy Document Analyzer + Coverage Clause Extractor + Restriction detection.

Two extractors run on every document:
  1. a transparent regex/rule extractor (always available, gives page evidence)
  2. an LLM extractor (Groq / Gemini) fed with retrieved, page-tagged chunks
The results are merged and VALIDATED (types, ranges, cross-check between the
two extractors). Disagreements are surfaced as "validation_notes" so the user
knows which values need manual verification.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from .llm import LLMClient, LLMError
from .prompts import render, system_prompt
from .rag import PolicyIndex
from .text_clean import split_sentences

log = logging.getLogger(__name__)

# --------------------------------------------------------------------- schema
FIELD_GROUPS: dict[str, dict[str, tuple[str, str]]] = {
    "overview": {
        "insurer_name": ("str", "Insurance company name"),
        "policy_name": ("str", "Product / plan name"),
        "uin": ("str", "IRDAI Unique Identification Number of the product"),
        "policy_type": ("str", "One of: Individual Health, Family Floater, Senior Citizen, Group Health, Top-up Health, Critical Illness"),
        "sum_insured": ("int", "Sum insured in rupees (if one value is stated in a schedule/illustration; else null)"),
        "sum_insured_options": ("str", "Sum insured options/range offered"),
        "annual_premium": ("int", "Annual premium in rupees if stated; else null"),
        "policy_term_years": ("float", "Policy period in years"),
        "family_members_covered": ("str", "Who can be covered (self, spouse, children, parents...)"),
        "age_band": ("str", "Entry age / age limits"),
        "coverage_category": ("str", "Indemnity hospitalisation / benefit / top-up etc."),
    },
    "coverage": {
        "inpatient_hospitalization": ("bool", "In-patient hospitalisation covered"),
        "pre_hospitalization_days": ("int", "Days of pre-hospitalisation expenses covered"),
        "post_hospitalization_days": ("int", "Days of post-hospitalisation expenses covered"),
        "day_care_covered": ("bool", "Day care procedures covered"),
        "ambulance_cover": ("str", "Ambulance cover and its limit"),
        "domiciliary_covered": ("bool", "Domiciliary (home) hospitalisation covered"),
        "restoration_benefit": ("bool", "Restoration / recharge / reinstatement of sum insured available"),
        "restoration_details": ("str", "Restoration conditions"),
        "no_claim_bonus": ("bool", "No claim bonus / cumulative bonus available"),
        "no_claim_bonus_details": ("str", "NCB / cumulative bonus details"),
        "maternity_covered": ("bool", "Maternity expenses covered"),
        "maternity_details": ("str", "Maternity waiting period / limits"),
        "newborn_covered": ("bool", "New born baby covered"),
        "child_coverage": ("bool", "Dependent children can be covered"),
        "child_coverage_details": ("str", "Child age limits / conditions"),
        "ayush_covered": ("bool", "AYUSH treatment covered"),
        "opd_covered": ("bool", "OPD / outpatient expenses covered"),
    },
    "restrictions": {
        "initial_waiting_period_days": ("int", "Initial waiting period in days for illnesses (first X days)"),
        "ped_waiting_period_months": ("int", "Pre-existing disease waiting period in months"),
        "specific_disease_waiting_period_months": ("int", "Specified disease/procedure waiting period in months"),
        "copayment_percentage": ("float", "Co-payment percentage (0 if the document says no co-payment)"),
        "copayment_conditions": ("str", "When the co-payment applies (all claims / age / zone / non-network)"),
        "room_rent_limit_present": ("bool", "Room rent is capped (% of SI, Rs per day or room category)"),
        "room_rent_limit": ("str", "The room rent limit wording"),
        "icu_limit_present": ("bool", "ICU / ICCU charges are capped"),
        "icu_limit": ("str", "The ICU limit wording"),
        "disease_sub_limits": ("list", "Disease/procedure-wise sub-limits, e.g. 'Cataract: 25% of SI or Rs 40,000'"),
        "permanent_exclusions": ("list", "Main permanent exclusion categories (short names)"),
        "temporary_exclusions": ("list", "Exclusions that apply only for a period (waiting-period based)"),
        "non_payable_items_mentioned": ("bool", "Document mentions a list of non-payable / consumable items"),
    },
    "claims_renewal": {
        "cashless_available": ("bool", "Cashless claim facility available"),
        "reimbursement_available": ("bool", "Reimbursement claim facility available"),
        "claim_intimation_timeline": ("str", "Time limit to intimate the claim"),
        "claim_documents_listed": ("bool", "List of claim documents is provided"),
        "claim_settlement_timeline": ("str", "Days within which claims are settled"),
        "renewal_conditions": ("str", "Renewal conditions (lifelong, age limits, etc.)"),
        "grace_period_days": ("int", "Grace period for renewal in days"),
        "portability_mentioned": ("bool", "Portability / migration with continuity is described"),
        "moratorium_mentioned": ("bool", "Moratorium period clause is present"),
    },
}

GROUP_QUERIES = {
    "overview": ["insurer company name policy name product UIN", "sum insured options premium policy period",
                 "eligibility entry age family floater individual members covered"],
    "coverage": ["in-patient hospitalisation expenses covered", "pre-hospitalisation post-hospitalisation days",
                 "day care ambulance domiciliary AYUSH", "restoration recharge reinstatement of sum insured",
                 "no claim bonus cumulative bonus", "maternity newborn baby dependent child cover", "OPD outpatient"],
    "restrictions": ["pre-existing diseases waiting period months", "specified disease procedure waiting period",
                     "first 30 days waiting period", "co-payment percentage claim", "room rent limit per day sum insured",
                     "ICU ICCU charges limit", "sub-limit cataract knee replacement limited to",
                     "permanent exclusions list", "non-payable items consumables"],
    "claims_renewal": ["cashless claim procedure network hospital", "reimbursement claim documents submit within days",
                       "claim settlement within days", "renewal grace period", "portability migration moratorium"],
}
GROUP_CATEGORIES = {
    "overview": ["premium", "general"], "coverage": ["coverage"],
    "restrictions": ["waiting_period", "exclusions", "copay_limits"],
    "claims_renewal": ["claims", "renewal"],
}

ALL_FIELDS = {k: v for g in FIELD_GROUPS.values() for k, v in g.items()}


def empty_extraction() -> dict:
    data: dict[str, Any] = {k: None for k in ALL_FIELDS}
    data.update(evidence={}, ambiguous_clauses=[], validation_notes=[], exclusion_count=None,
                extraction_method="rules")
    return data


# --------------------------------------------------------------------- sentences
def _sentences(blocks: list[dict]) -> list[dict]:
    """Split blocks into sentences, each tagged with page + section category/title."""
    out = []
    for b in blocks:
        para: list[str] = []
        page = None

        def flush():
            if para:
                text = " ".join(para)
                for s in split_sentences(text):
                    s = s.strip()
                    if len(s) > 8:
                        # section title is prepended for keyword matching only (not for quotes)
                        low = s.lower() if s == b["section_title"] else f"{b['section_title']} :: {s}".lower()
                        out.append({"text": s, "low": low, "page": page, "heading": s == b["section_title"] and len(s.split()) < 8
                                    and not re.search(r"\d+\s*(months|days|years|%)", s.lower()),
                                    "cat": b["section_category"], "title": b["section_title"]})

        for idx, (pg, line) in enumerate(b["lines"]):
            if page is not None and pg != page:
                flush()
                para = []
            page = pg
            para.append(line)
            if idx == 0 and line == b["section_title"]:  # keep the heading as its own "sentence"
                flush()
                para = []
        flush()
    return out


NEG = re.compile(r"\b(not covered|not available|not payable|excluded|no cover|shall not|is not|are not|nil)\b")
_NUM_WORD = r"(?:\s*\([a-z\s-]+\))?"


def _to_months(num: str, unit: str) -> int:
    n = float(num)
    return int(round(n * 12)) if unit.lower().startswith("year") else int(round(n))


def _rupees(num: str, unit: str | None) -> int:
    n = float(num.replace(",", ""))
    unit = (unit or "").lower()
    if unit.startswith(("lakh", "lac")):
        n *= 100_000
    elif unit.startswith("crore"):
        n *= 10_000_000
    return int(n)


def _around(text: str, pattern: str, width: int = 200) -> str:
    """Snippet of `text` starting near the first match of pattern (avoids table headers / preambles)."""
    m = re.search(pattern, text, re.I)
    start = max(0, m.start() - 20) if m else 0
    snippet = text[start:start + width]
    return ("\u2026" if start else "") + snippet.strip()


def _ev(sent: dict) -> dict:
    words = sent["text"].split()
    return {"page": sent["page"], "quote": " ".join(words[:30])}


DISEASE_TERMS = ["cataract", "knee", "hip", "joint replacement", "hernia", "hysterectomy", "stone",
                 "kidney", "gall", "sinus", "tonsil", "piles", "haemorrhoid", "prostate", "cancer",
                 "dialysis", "modern treatment", "robotic", "bariatric", "psychiatric", "mental",
                 "hiv", "cardiac", "heart", "spine", "varicose", "glaucoma", "dental", "ent",
                 "balloon sinuplasty", "intra vitreal", "stem cell", "oral chemotherapy", "immunotherapy"]
_SUBLIMIT_GROUP = {t: "modern treatments" for t in
                   ["modern treatment", "robotic", "oral chemotherapy", "immunotherapy", "stem cell",
                    "intra vitreal", "balloon sinuplasty"]}
_SUBLIMIT_GROUP.update({"cardiac": "heart", "haemorrhoid": "piles", "psychiatric": "mental",
                        "joint replacement": "knee"})
AMBIGUOUS_PHRASES = ["sole discretion", "at its discretion", "reasonable and customary",
                     "reasonable and necessary", "as deemed fit", "may at its option",
                     "from time to time", "as specified in the policy schedule",
                     "as mentioned in the policy schedule", "subject to such", "if applicable",
                     "as applicable", "medically necessary", "customary charges"]


# --------------------------------------------------------------------- rule extractor
def regex_extract(blocks: list[dict], pages: list[dict], raw_pages: list[dict] | None = None) -> dict:
    data = empty_extraction()
    ev = data["evidence"]
    all_sents = _sentences(blocks)
    sents = [s for s in all_sents if not s["heading"]]  # headings only add context via s["low"]
    non_excl = [s for s in sents if s["cat"] not in ("exclusions", "definitions")]
    body = [s for s in sents if s["cat"] != "definitions" and " means " not in s["low"][:80]]

    def first(pred, pool=body):
        return next((s for s in pool if pred(s)), None)

    # ---------------- overview
    raw_pages = raw_pages or pages
    page1 = "\n".join(p["text"] for p in raw_pages[:2])
    for m in re.finditer(r"((?:[\w&.\-]+\s+){1,6}?(?:General\s+|Health\s+|Allied\s+)*Insurance\s+"
                         r"Co(?:mpany|\.)?(?:\s+of\s+India)?\s*(?:Ltd\.?|Limited))", page1, re.I):
        words = m.group(1).split()
        while words and not words[0][0].isupper():  # drop "issued by", "the" ...
            words.pop(0)
        while words and words[0].lower() in ("this", "the", "by", "policy", "is", "a"):
            words.pop(0)
        if len(words) >= 3:
            name = " ".join(words)
            data["insurer_name"] = name.title() if name.isupper() else name
            break
    m = re.search(r"\b([A-Z]{7}\d{5}V\d{6})\b", " ".join(p["text"] for p in raw_pages))
    if m:
        data["uin"] = m.group(1)
    for line in page1.splitlines()[:25]:
        clean = line
        if data["insurer_name"]:
            clean = clean.replace(data["insurer_name"], "")
        clean = re.sub(r"^\s*policy wordings?\s*[-\u2013:]*\s*", "", clean, flags=re.I)
        clean = re.sub(r"\s*[-\u2013]\s*policy wordings?\s*$", "", clean, flags=re.I).strip(" ,:-\u2013")
        low = clean.lower()
        clean = re.sub(r"^\s*uin\s*:\s*", "", clean, flags=re.I).strip(" ,:-\u2013")
        low = clean.lower()
        if 4 < len(clean) < 80 and re.search(r"(policy|plan|insurance|health|care|secure|optima)\b", low) \
                and not re.search(r"company|limited|\bltd\b|^preamble|contract of|reg\. no|cin:|\d{5}", low) \
                and low not in ("policy wording", "policy wordings", "prospectus", "customer information sheet"):
            data["policy_name"] = clean
            break
    full_low = " ".join(p["text"] for p in pages).lower()
    if "senior citizen" in full_low[:6000]:
        data["policy_type"] = "Senior Citizen"
    elif re.search(r"super top[- ]?up|top[- ]?up", full_low[:6000]):
        data["policy_type"] = "Top-up Health"
    elif "critical illness" in full_low[:3000]:
        data["policy_type"] = "Critical Illness"
    elif re.search(r"\bgroup\b", full_low[:3000]):
        data["policy_type"] = "Group Health"
    elif "floater" in full_low:
        data["policy_type"] = "Family Floater"
    else:
        data["policy_type"] = "Individual Health"

    s = first(lambda s: "sum insured" in s["low"] and re.search(r"(rs\.?|inr|\u20b9)\s*[\d,]{5,}", s["low"])
              and not re.search(r"%\s*of\s*(the\s*)?sum insured|up ?to|maximum|per day|limit", s["low"]))
    if s:
        amounts = re.findall(r"(?:rs\.?|inr|\u20b9)?\s*([\d,.]+)\s*(lakhs?|lacs?|crores?)", s["low"]) + \
            re.findall(r"(?:rs\.?|inr|\u20b9)\s*([\d,]{5,})", s["low"])
        if len(amounts) >= 2 or re.search(r"options?|/\s*\d|\bto\b\s*(rs|inr|\d)", s["low"]):
            data["sum_insured_options"] = s["text"][:200]  # a menu of options, not this policy's value
        else:
            m = re.search(r"(?:rs\.?|inr|\u20b9)\s*([\d,]{5,}(?:\.\d+)?)\s*(lakhs?|lacs?|crores?)?", s["low"])
            data["sum_insured"] = _rupees(m.group(1), m.group(2))
            ev["sum_insured"] = _ev(s)
    s = first(lambda s: re.search(r"(annual|total|yearly)\s+premium", s["low"])
              and re.search(r"(rs\.?|inr|\u20b9)\s*[\d,]{3,}", s["low"]))
    if s:
        m = re.search(r"(?:rs\.?|inr|\u20b9)\s*([\d,]{3,})", s["low"])
        data["annual_premium"] = _rupees(m.group(1), None)
        ev["annual_premium"] = _ev(s)
    s = first(lambda s: re.search(r"policy (period|term)", s["low"]) and re.search(r"\d\s*years?", s["low"]))
    if s:
        data["policy_term_years"] = float(re.search(r"(\d)\s*years?", s["low"]).group(1))

    # ---------------- coverage
    def days_after(keyword_re):
        s = first(lambda s: re.search(keyword_re, s["low"]) and re.search(r"\d{2,3}\s*days", s["low"]), non_excl)
        if s:
            seg = re.split(keyword_re, s["low"], maxsplit=1)[-1]
            m = re.search(r"(\d{2,3})" + _NUM_WORD + r"\s*days", seg) or re.search(r"(\d{2,3})\s*days", s["low"])
            return int(m.group(1)), s
        return None, None

    for field, kw in (("pre_hospitalization_days", r"pre[- ]?hospitali[sz]ation"),
                      ("post_hospitalization_days", r"post[- ]?hospitali[sz]ation")):
        val, s = days_after(kw)
        if val:
            data[field], ev[field] = val, _ev(s)

    excl_sents = [s for s in all_sents if s["cat"] == "exclusions"]

    def presence(field, pattern, pool=non_excl):
        s = first(lambda s: re.search(pattern, s["low"]) and not re.search(r"add.?on|optional", s["low"]), pool)
        if s:
            data[field] = not bool(NEG.search(s["low"]))
            ev[field] = _ev(s)
        else:  # only mentioned inside the exclusions section -> not covered
            s = first(lambda s: re.search(pattern, s["low"]), excl_sents)
            if s:
                data[field], ev[field] = False, _ev(s)

    presence("inpatient_hospitalization", r"in-?patient|hospitali[sz]ation expenses")
    presence("day_care_covered", r"day ?care (procedure|treatment)")
    presence("domiciliary_covered", r"domiciliary")
    presence("ayush_covered", r"ayush")
    presence("opd_covered", r"\bopd\b|out-?patient")
    s = first(lambda s: "ambulance" in s["low"], non_excl)
    if s:
        data["ambulance_cover"] = s["text"][:160]
        ev["ambulance_cover"] = _ev(s)
    s = first(lambda s: re.search(r"(restor\w*|recharg\w*|reinstat\w*|refill\w*)\W+(?:\w+\W+){0,4}sum insured|"
                                  r"sum insured\W+(?:\w+\W+){0,4}(restor|recharg|reinstat|refill)|"
                                  r"secure benefit|restore benefit|restoration benefit", s["low"]), body)
    if s:
        data["restoration_benefit"] = not bool(re.search(r"not available|not applicable|no restor", s["low"]))
        data["restoration_details"] = s["text"][:220]
        ev["restoration_benefit"] = _ev(s)
    s = first(lambda s: re.search(r"no claim bonus|cumulative bonus", s["low"]), body)
    if s:
        data["no_claim_bonus"] = not bool(re.search(r"not available|not applicable", s["low"]))
        data["no_claim_bonus_details"] = s["text"][:220]
        ev["no_claim_bonus"] = _ev(s)
    s = first(lambda s: "maternity" in s["low"] and not NEG.search(s["low"])
              and not re.search(r"add.?on|optional|parenthood", s["low"]), non_excl)
    if s:
        data["maternity_covered"] = True
        data["maternity_details"] = s["text"][:220]
        ev["maternity_covered"] = _ev(s)
    else:
        s = first(lambda s: "maternity" in s["low"] or "pregnan" in s["low"], sents)
        if s and (s["cat"] == "exclusions" or NEG.search(s["low"])):
            data["maternity_covered"] = False
            ev["maternity_covered"] = _ev(s)
    s = first(lambda s: re.search(r"new ?born", s["low"]) and not NEG.search(s["low"]), non_excl)
    if s:
        data["newborn_covered"] = True
        ev["newborn_covered"] = _ev(s)
    else:
        s = first(lambda s: re.search(r"new ?born", s["low"]), sents)
        if s and (s["cat"] == "exclusions" or NEG.search(s["low"])):
            data["newborn_covered"], ev["newborn_covered"] = False, _ev(s)
    s = first(lambda s: re.search(r"dependent child|children|\bchild\b", s["low"])
              and not NEG.search(s["low"]), non_excl)
    if s:
        data["child_coverage"] = True
        data["child_coverage_details"] = s["text"][:220]
        ev["child_coverage"] = _ev(s)

    # ---------------- waiting periods
    def months_in(*preds):
        """Try predicates in order (strongest first). Multi-option values like '36 / 24 / 12 months'
        or '24/36 months' return the LONGEST option (conservative) and are flagged as unclear."""
        for pred in preds:
            anchor = None
            if isinstance(pred, tuple):  # (anchor regex, predicate): read the value right after the anchor
                anchor, pred = pred
            for s in body:
                if pred(s) and "prior to" not in s["low"]:
                    low = s["low"]
                    if anchor:
                        a = re.search(anchor, low)
                        low = low[a.end():a.end() + 160] if a else low
                    tier = re.search(r"((?:\d{1,2}\s*/\s*)+\d{1,2})\s*(months|years)", low)
                    if tier:
                        opts = [int(x) for x in re.findall(r"\d{1,2}", tier.group(1))]
                        data["ambiguous_clauses"].append(
                            f"p.{s['page']}: waiting period has options '{tier.group(0)}' - depends on plan/"
                            f"schedule or condition; longest used for scoring")
                        return _to_months(str(max(opts)), tier.group(2)), s
                    m = re.search(r"(\d{1,2})" + _NUM_WORD + r"\s*(months|years)", low)
                    if m:
                        return _to_months(m.group(1), m.group(2)), s
        return None, None

    wait_ctx = r"waiting|expiry|excluded until|continuous coverage"
    val, s = months_in((r"excl\s?-?01", lambda s: re.search(r"excl\s?-?01", s["low"]) and re.search(wait_ctx, s["low"])),
                       lambda s: re.search(r"pre-?\s?existing", s["low"]) and re.search(wait_ctx, s["low"])
                       and not re.search(r"modif|optional|add.?on", s["low"]))
    if val is not None:
        data["ped_waiting_period_months"], ev["ped_waiting_period_months"] = val, _ev(s)
    val, s = months_in((r"excl\s?-?02", lambda s: re.search(r"excl\s?-?02", s["low"]) and re.search(wait_ctx, s["low"])),
                       lambda s: re.search(r"specified (disease|illness)|specific (disease|illness)|"
                                           r"listed (conditions|diseases)", s["low"])
                       and re.search(wait_ctx, s["low"]) and not re.search(r"modif|optional|add.?on", s["low"]))
    if val is not None:
        data["specific_disease_waiting_period_months"] = val
        ev["specific_disease_waiting_period_months"] = _ev(s)
    s = first(lambda s: re.search(r"excl\s?-?03|initial waiting", s["low"]) and re.search(r"\d{2,3}\s*days", s["low"])) \
        or first(lambda s: re.search(r"first \d{2,3}" + _NUM_WORD + r"\s*days", s["low"])
                 and re.search(r"illness|polic", s["low"]) and "trip" not in s["low"])
    if s:
        m = re.search(r"(\d{2,3})" + _NUM_WORD + r"\s*days", s["low"])
        if m:
            data["initial_waiting_period_days"], ev["initial_waiting_period_days"] = int(m.group(1)), _ev(s)

    # ---------------- co-payment
    no_copay = first(lambda s: re.search(r"no co-?\s?pay|co-?\s?payment\s*(:|-)?\s*(nil|not applicable)", s["low"]))
    s = first(lambda s: re.search(r"co-?\s?pay", s["low"]) and re.search(r"\d{1,2}(\.\d+)?\s*%", s["low"]))
    if no_copay:  # general rule is "no co-pay"; any % found is a conditional co-pay
        data["copayment_percentage"] = 0.0
        data["copayment_conditions"] = (f"Conditional: {s['text'][:200]}" if s else no_copay["text"][:200])
        ev["copayment_percentage"] = _ev(no_copay)
    elif s:
        seg = re.split(r"co-?\s?pay\w*", s["low"], maxsplit=1)
        m = re.search(r"(\d{1,2}(?:\.\d+)?)\s*%", seg[-1]) or re.search(r"(\d{1,2}(?:\.\d+)?)\s*%", s["low"])
        data["copayment_percentage"] = float(m.group(1))
        data["copayment_conditions"] = s["text"][:220]
        ev["copayment_percentage"] = _ev(s)


    # ---------------- room rent / ICU
    no_cap = re.compile(r"no (capping|cap|limit|sub-?limit|restriction)|any room|without any (cap|limit)|"
                        r"at actuals?")
    optional = re.compile(r"modif\w*|optional|add.?on|on availing")

    def limit_decision(pattern):
        """Walk every sentence about the limit until one states a cap or 'no cap / at actuals'."""
        for s in body:
            if not re.search(pattern, s["low"]) or optional.search(s["text"].lower()):
                continue
            if no_cap.search(s["low"]) or (" actual" in s["low"] and "icu" in pattern):
                return False, s
            if re.search(r"\d\s*%|rs\.?\s*[\d,]+|\u20b9|per day|single private|shared room|twin sharing",
                         s["low"]):
                return True, s
        return None, None

    present, s = limit_decision(r"room rent|room,? boarding")
    if s:
        data["room_rent_limit_present"] = present
        data["room_rent_limit"] = _around(s["text"], r"room rent|room,? boarding")
        ev["room_rent_limit_present"] = {"page": s["page"], "quote": data["room_rent_limit"][:160]}
        if re.search(r"unless otherwise specified|as per plan|basis plan|single private room|shared room",
                     s["low"]) and re.search(r"actual", s["low"]):
            data["ambiguous_clauses"].append(
                f"p.{s['page']}: room rent limit depends on the plan / Policy Schedule - check your schedule")
    present, s = limit_decision(r"\bicu\b|iccu|intensive care")
    if s:
        data["icu_limit_present"] = present
        data["icu_limit"] = _around(s["text"], r"\bicu\b|iccu|intensive care")
        ev["icu_limit_present"] = _ev(s)

    # ---------------- sub-limits
    subl = []
    for s in body:
        if re.search(r"sub-?limit|limited to|limit of|up to|upto|maximum of|capped", s["low"]) and \
                not re.search(r"\bicu\b|iccu|intensive c|room rent|ambulance", s["low"]) and \
                re.search(r"\d\s*%|rs\.?\s*[\d,]+|\u20b9", s["low"]) and "waiting" not in s["low"]:
            for term in DISEASE_TERMS:
                group = _SUBLIMIT_GROUP.get(term, term)
                if re.search(rf"\b{re.escape(term)}", s["low"]) and group not in [t for t, _ in subl]:
                    subl.append((group, s))
    data["disease_sub_limits"] = [f"{t.title()}: {s['text'][:120]}" for t, s in subl]
    if subl:
        ev["disease_sub_limits"] = _ev(subl[0][1])

    # ---------------- exclusions
    codes = set(re.findall(r"excl\s?-?(\d{2})", full_low))
    excl_items = []
    for b in blocks:
        if b["section_category"] == "exclusions":
            title = b["section_title"]
            if not re.search(r"^(section\s+\w+\s*[-:]?\s*)?\d*\.?\s*exclusions?\b", title, re.I) and \
                    re.match(r"^([a-z]\.)?\d", title, re.I):
                excl_items.append(title)
            for _, line in b["lines"][1:]:
                if re.match(r"^(\d{1,2}[.)]|[a-z][.)]|[ivx]+[.)]|\u2022|-)\s+\S", line, re.I) and len(line) < 200:
                    excl_items.append(line)
    if len(codes) >= 5:
        uncoded = [t for t in excl_items if not re.search(r"excl\s?-?\d|non-?payable", t, re.I)]
        data["exclusion_count"] = len(codes) + min(len(uncoded), 10)
    else:
        data["exclusion_count"] = len(excl_items) or None
    data["permanent_exclusions"] = [re.sub(r"^\S+\s+|\(?\s*code\s*[-:]?\s*excl\s?\d+\s*\)?:?", "", l,
                                           flags=re.I).strip(" :-")[:90] for l in excl_items[:25]]
    data["non_payable_items_mentioned"] = bool(re.search(r"non-?payable|consumables|optional items", full_low))
    tmp = [s["text"][:120] for s in body if re.search(r"waiting period|excluded until", s["low"])]
    data["temporary_exclusions"] = tmp[:6]

    # ---------------- claims & renewal
    claim_sents = [s for s in sents if s["cat"] == "claims"] or sents
    data["cashless_available"] = any("cashless" in s["low"] for s in claim_sents) or None
    data["reimbursement_available"] = any("reimbursement" in s["low"] for s in claim_sents) or None
    s = first(lambda s: re.search(r"(intimat|notif|notice)", s["low"]) and re.search(r"within \d+\s*(hours|days)", s["low"])
              and not re.search(r"settle|tat|turn around", s["low"]), claim_sents)
    if s:
        data["claim_intimation_timeline"] = re.search(r"within \d+\s*(hours|days)[^.;,]{0,60}", s["low"]).group(0)
        ev["claim_intimation_timeline"] = _ev(s)
    data["claim_documents_listed"] = any(re.search(r"documents?", s["low"]) and
                                         re.search(r"discharge summary|bills|claim form|submit", s["low"])
                                         for s in claim_sents) or None
    s = first(lambda s: re.search(r"settle|payment of claim|pay or reject", s["low"])
              and re.search(r"within \d+\s*days", s["low"]), claim_sents)
    if s:
        data["claim_settlement_timeline"] = re.search(r"within \d+\s*days", s["low"]).group(0)
        ev["claim_settlement_timeline"] = _ev(s)
    s = first(lambda s: "grace period" in s["low"] and re.search(r"\d+\s*days", s["low"]), sents)
    if s:
        data["grace_period_days"] = int(re.search(r"(\d+)" + _NUM_WORD + r"\s*days", s["low"]).group(1))
        ev["grace_period_days"] = _ev(s)
    s = first(lambda s: re.search(r"renewal|renewable", s["low"]) and re.search(r"lifelong|life ?long|cannot be denied|shall not be denied|ordinarily", s["low"]), sents)
    if s:
        data["renewal_conditions"] = s["text"][:200]
    data["portability_mentioned"] = "portab" in full_low or None
    data["moratorium_mentioned"] = "moratorium" in full_low or None

    # ---------------- ambiguity
    amb = []
    for phrase in AMBIGUOUS_PHRASES:
        s = first(lambda s, p=phrase: p in s["low"])
        if s:
            amb.append(f"p.{s['page']}: \u201c{s['text'][:150]}\u201d")
    data["ambiguous_clauses"] = amb + data["ambiguous_clauses"]
    return data


# --------------------------------------------------------------------- LLM extractor
def _context_for(index: PolicyIndex, group: str, max_chars: int = 10000) -> str:
    seen, parts, size = set(), [], 0
    for q in GROUP_QUERIES[group]:
        for chunk, _ in index.search(q, k=4, categories=GROUP_CATEGORIES[group]):
            if chunk.chunk_id in seen:
                continue
            block = f"[p.{chunk.page_start} | {chunk.section_title}]\n{chunk.text}\n"
            if size + len(block) > max_chars:
                break
            seen.add(chunk.chunk_id)
            parts.append(block)
            size += len(block)
    if group == "overview" and index.chunks:  # page 1 usually carries insurer / plan name
        first = [c for c in index.chunks if c.page_start <= 1][:2]
        parts = [f"[p.1 | {c.section_title}]\n{c.text}\n" for c in first if c.chunk_id not in seen] + parts
    return "\n".join(parts)


def llm_extract(index: PolicyIndex, llm: LLMClient) -> dict:
    result: dict[str, Any] = {"evidence": {}, "ambiguous_clauses": []}
    for group, fields in FIELD_GROUPS.items():
        spec = "\n".join(f'- "{k}" ({t}): {d}' for k, (t, d) in fields.items())
        prompt = render("extraction", fields=spec, context=_context_for(index, group))
        try:
            out = llm.chat_json(system_prompt(), prompt, max_tokens=2500)
        except LLMError as exc:
            log.warning("LLM extraction failed for %s: %s", group, exc)
            continue
        for k in fields:
            if k in out:
                result[k] = out[k]
        if isinstance(out.get("evidence"), dict):
            result["evidence"].update(out["evidence"])
        if isinstance(out.get("ambiguous_clauses"), list):
            result["ambiguous_clauses"].extend(str(x) for x in out["ambiguous_clauses"])
    return result


# --------------------------------------------------------------------- validation
RANGES = {
    "copayment_percentage": (0, 100), "ped_waiting_period_months": (0, 96),
    "specific_disease_waiting_period_months": (0, 96), "initial_waiting_period_days": (0, 365),
    "pre_hospitalization_days": (0, 365), "post_hospitalization_days": (0, 365),
    "grace_period_days": (0, 120), "sum_insured": (10_000, 1_000_000_000),
    "annual_premium": (100, 5_000_000), "policy_term_years": (1, 3),  # retail health: 1-3 year terms
}
CROSS_CHECK = ["copayment_percentage", "ped_waiting_period_months", "specific_disease_waiting_period_months",
               "initial_waiting_period_days", "room_rent_limit_present", "icu_limit_present",
               "restoration_benefit", "maternity_covered", "newborn_covered", "opd_covered",
               "domiciliary_covered", "pre_hospitalization_days", "post_hospitalization_days"]


def _coerce(value, typ, field: str = ""):
    if value is None or (isinstance(value, str) and value.strip().lower() in
                         ("", "null", "none", "n/a", "na", "not mentioned", "not found", "unknown")):
        return None
    try:
        if typ in ("int", "float"):
            if isinstance(value, str):
                low = value.lower().replace(",", "")
                m = re.search(r"(\d+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|years?|months?)?", low)
                if not m:
                    return None
                num = float(m.group(1))
                unit = m.group(2) or ""
                if unit.startswith(("lakh", "lac")):
                    num *= 100_000
                elif unit.startswith("crore"):
                    num *= 10_000_000
                elif unit.startswith("year") and field.endswith("_months"):
                    num *= 12
                elif unit.startswith("month") and field.endswith("_days"):
                    num *= 30
                value = num
            return int(round(float(value))) if typ == "int" else float(value)
        if typ == "bool":
            if isinstance(value, bool):
                return value
            low = str(value).strip().lower()
            if low in ("true", "yes", "covered", "available", "1"):
                return True
            if low in ("false", "no", "not covered", "excluded", "not available", "0"):
                return False
            return None
        if typ == "list":
            if isinstance(value, list):
                return [str(v) for v in value if str(v).strip()]
            return [str(value)]
        return str(value).strip()
    except (ValueError, TypeError):
        return None


def validate_extraction(data: dict) -> dict:
    notes = data.setdefault("validation_notes", [])
    for field, (typ, _) in ALL_FIELDS.items():
        data[field] = _coerce(data.get(field), typ, field)
        if field in RANGES and data[field] is not None:
            lo, hi = RANGES[field]
            if not lo <= data[field] <= hi:
                notes.append(f"{field}={data[field]} is outside the plausible range [{lo}, {hi}] - ignored.")
                data[field] = None
    if data.get("sum_insured") and data.get("annual_premium"):
        if data["annual_premium"] / data["sum_insured"] > 0.5:
            notes.append("Premium looks too high relative to sum insured - please check both values.")
    if data.get("room_rent_limit_present") is None and data.get("room_rent_limit"):
        data["room_rent_limit_present"] = True
    if data.get("exclusion_count") is None and data.get("permanent_exclusions"):
        data["exclusion_count"] = len(data["permanent_exclusions"])
    return data


def merge_extractions(rule: dict, llm_out: dict | None) -> dict:
    merged = dict(rule)
    if not llm_out:
        return validate_extraction(merged)
    merged["extraction_method"] = "llm+rules"
    notes = merged.setdefault("validation_notes", [])
    for field, (typ, _) in ALL_FIELDS.items():
        lv = _coerce(llm_out.get(field), typ, field)
        rv = rule.get(field)
        if lv is None:
            continue
        if field in CROSS_CHECK and rv is not None and lv != rv:
            page = rule["evidence"].get(field, {}).get("page", "?")
            notes.append(f"{field}: LLM read '{lv}', rule extractor found '{rv}' (p.{page}) - "
                         f"kept the rule value (base wording); verify in the document.")
            continue  # base-policy evidence wins over LLM reads of add-ons / plan options
        if field in ("disease_sub_limits", "permanent_exclusions", "temporary_exclusions"):
            merged[field] = lv if len(lv) >= len(rv or []) else rv
        else:
            merged[field] = lv
    for field, e in (llm_out.get("evidence") or {}).items():
        if isinstance(e, dict) and field in ALL_FIELDS:
            merged["evidence"][field] = e
    seen = set()
    merged["ambiguous_clauses"] = [a for a in rule["ambiguous_clauses"] + llm_out.get("ambiguous_clauses", [])
                                   if not (a[:60] in seen or seen.add(a[:60]))][:15]
    return validate_extraction(merged)


def extract_clauses(blocks: list[dict], pages: list[dict], index: PolicyIndex,
                    llm: LLMClient | None = None, raw_pages: list[dict] | None = None) -> dict:
    rule = regex_extract(blocks, pages, raw_pages)
    llm_out = llm_extract(index, llm) if (llm and llm.available) else None
    return merge_extractions(rule, llm_out)