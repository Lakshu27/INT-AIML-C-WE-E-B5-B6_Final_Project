"""Feature Builder - converts extracted clauses into the numerical features used
by the Coverage Clarity / Risk scoring model.

The definitions are aligned with coverwise_synthetic_coverage_risk_dataset_500.csv.
Analysis of that dataset showed `risk_flags_count` is (corr 0.95) the count of
the 10 transparent flags in MODEL_FLAG_RULES below, so the same rules are used
at inference time - training and serving features stay consistent.
"""
from __future__ import annotations

from . import config

# median-ish values from the synthetic dataset used ONLY when a value is missing;
# every imputed field is reported back as "unclear" to the user.
IMPUTE = {
    "sum_insured": 1_000_000, "annual_premium": 24_350, "waiting_period_months": 24,
    "pre_existing_disease_waiting_period_months": 36, "copayment_percentage": 0,
    "exclusion_count": 13, "disease_sub_limit_count": 1,
}
TRAIN_RANGE = {  # clip to the training distribution so the model is not extrapolating
    "sum_insured": (300_000, 5_000_000), "annual_premium": (1_800, 288_400),
    "premium_to_coverage_ratio": (0.006, 0.062), "waiting_period_months": (0, 48),
    "pre_existing_disease_waiting_period_months": (0, 48), "copayment_percentage": (0, 30),
    "disease_sub_limit_count": (0, 6), "exclusion_count": (3, 22),
    "claim_process_clarity_score": (35, 100), "ambiguous_clause_count": (0, 6),
    "risk_flags_count": (0, 10),
}

# (feature-level test, short flag, why it matters)
MODEL_FLAG_RULES = [
    ("pre_existing_disease_waiting_period_months", lambda v: v >= 36,
     "Long pre-existing disease waiting period (>= 36 months)",
     "Claims for conditions you already have may not be paid for several years."),
    ("copayment_percentage", lambda v: v >= 10,
     "Co-payment of 10% or more",
     "You pay this share of every admissible claim from your own pocket."),
    ("room_rent_limit_present", lambda v: v == 1,
     "Room rent is capped",
     "Choosing a costlier room can reduce the whole claim proportionately, not just the room charge."),
    ("icu_limit_present", lambda v: v == 1,
     "ICU charges are capped",
     "Long ICU stays can exceed the cap and the excess is paid by you."),
    ("disease_sub_limit_count", lambda v: v >= 3,
     "Several disease / procedure sub-limits (>= 3)",
     "Common surgeries (e.g. cataract, joint replacement) may be paid only up to a fixed amount."),
    ("exclusion_count", lambda v: v >= 15,
     "Large number of exclusions (>= 15)",
     "More situations where claims are simply not payable."),
    ("ambiguous_clause_count", lambda v: v >= 3,
     "Several vague / discretionary clauses (>= 3)",
     "Vague wording leaves room for disputes at claim time."),
    ("restoration_benefit_present", lambda v: v == 0,
     "No restoration / recharge of sum insured",
     "If the sum insured is used up, there is no automatic top-up in the same year."),
    ("claim_process_clarity_score", lambda v: v < 60,
     "Claim process is not clearly described",
     "Unclear timelines/documents increase the chance of claim delays or rejection."),
    ("waiting_period_months", lambda v: v >= 24,
     "Long specified-disease waiting period (>= 24 months)",
     "Listed illnesses/surgeries are not covered for the first 2+ years."),
]


def claim_process_clarity(ext: dict) -> int:
    """35 base + 13 for each clearly described claim element (max 100)."""
    score = 35
    for key in ("cashless_available", "reimbursement_available", "claim_intimation_timeline",
                "claim_documents_listed", "claim_settlement_timeline"):
        if ext.get(key):
            score += 13
    return min(score, 100)


def _b(value) -> int:
    return 1 if value is True else 0


def build_features(ext: dict, overrides: dict | None = None) -> tuple[dict, list[str]]:
    """Return (features, unclear_fields). overrides = user-entered premium / sum insured."""
    overrides = overrides or {}
    unclear: list[str] = []

    def num(field_value, key, label):
        if field_value is None:
            unclear.append(label)
            return IMPUTE[key]
        return field_value

    si = overrides.get("sum_insured") or ext.get("sum_insured")
    prem = overrides.get("annual_premium") or ext.get("annual_premium")
    f = {
        "policy_type": ext.get("policy_type") if ext.get("policy_type") in config.POLICY_TYPES
        else "Individual Health",
        "sum_insured": num(si, "sum_insured", "Sum insured"),
        "annual_premium": num(prem, "annual_premium", "Annual premium"),
        "waiting_period_months": num(ext.get("specific_disease_waiting_period_months"),
                                     "waiting_period_months", "Specified-disease waiting period"),
        "pre_existing_disease_waiting_period_months": num(
            ext.get("ped_waiting_period_months"), "pre_existing_disease_waiting_period_months",
            "Pre-existing disease waiting period"),
        "copayment_percentage": num(ext.get("copayment_percentage"), "copayment_percentage", "Co-payment"),
        "room_rent_limit_present": _b(ext.get("room_rent_limit_present")),
        "icu_limit_present": _b(ext.get("icu_limit_present")),
        "disease_sub_limit_count": len(ext.get("disease_sub_limits") or []),
        "exclusion_count": num(ext.get("exclusion_count"), "exclusion_count", "Exclusion list"),
        "restoration_benefit_present": _b(ext.get("restoration_benefit")),
        "no_claim_bonus_present": _b(ext.get("no_claim_bonus")),
        "child_coverage_present": _b(ext.get("child_coverage")),
        "maternity_coverage_present": _b(ext.get("maternity_covered")),
        "claim_process_clarity_score": claim_process_clarity(ext),
        "ambiguous_clause_count": len(ext.get("ambiguous_clauses") or []),
    }
    for key, label in (("room_rent_limit_present", "Room rent limit"), ("icu_limit_present", "ICU limit"),
                       ("restoration_benefit", "Restoration benefit")):
        if ext.get(key) is None:
            unclear.append(label)
    f["premium_to_coverage_ratio"] = round(f["annual_premium"] / max(f["sum_insured"], 1), 4)
    for k, (lo, hi) in TRAIN_RANGE.items():
        if k in f:
            f[k] = min(max(f[k], lo), hi)
    f["risk_flags_count"] = len(model_flags(f))
    return f, unclear


def model_flags(f: dict) -> list[dict]:
    flags = []
    for feat, test, flag, why in MODEL_FLAG_RULES:
        if feat in f and test(f[feat]):
            flags.append({"flag": flag, "why": why, "feature": feat, "value": f[feat]})
    return flags
