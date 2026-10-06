"""Central configuration for CoverWise AI.

All secrets come from environment variables (.env locally, Secrets on
Hugging Face Spaces / AWS). Nothing secret is ever hard-coded.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional
    load_dotenv = None

ROOT_DIR = Path(__file__).resolve().parent.parent
if load_dotenv:
    load_dotenv(ROOT_DIR / ".env")



def _env(name: str, default: str = "") -> str:
    """Read an env var, tolerating inline comments ('groq  # note') and stray quotes/spaces,
    which docker --env-file and some hosting UIs keep as part of the value."""
    value = os.getenv(name, default) or default
    if " #" in value or "\t#" in value:
        value = value.split("#", 1)[0]
    return value.strip().strip('"').strip("'").strip()


DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SAMPLE_POLICY_DIR = DATA_DIR / "sample_policies"
EVAL_DIR = DATA_DIR / "eval"
MODELS_DIR = ROOT_DIR / "models"
PROMPTS_DIR = ROOT_DIR / "prompts"
REPORTS_DIR = ROOT_DIR / "reports"

SYNTHETIC_DATASET = RAW_DATA_DIR / "coverwise_synthetic_coverage_risk_dataset_500.csv"
QA_TESTSET = RAW_DATA_DIR / "coverwise_policy_qa_risk_testset_60.csv"

# ---------------- LLM ----------------
# LLM_PROVIDER: "groq" | "gemini" | "none"  ("none" = offline rule-based mode)
LLM_PROVIDER = _env("LLM_PROVIDER", "groq").lower()
GROQ_API_KEY = _env("GROQ_API_KEY", "")
GROQ_MODEL = _env("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_FALLBACK_MODEL = _env("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")  # used when the daily limit is hit
GEMINI_API_KEY = _env("GEMINI_API_KEY", "")
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE = float(_env("LLM_TEMPERATURE", "0.1"))

# ---------------- RAG ----------------
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
USE_EMBEDDINGS = _env("USE_EMBEDDINGS", "1") == "1"  # falls back to TF-IDF if unavailable
CHUNK_SIZE = int(_env("CHUNK_SIZE", "1000"))         # characters
CHUNK_OVERLAP = int(_env("CHUNK_OVERLAP", "150"))
TOP_K = int(_env("TOP_K", "5"))
MIN_RETRIEVAL_SCORE = float(_env("MIN_RETRIEVAL_SCORE", "0.12"))

# ---------------- OCR ----------------
ENABLE_OCR = _env("ENABLE_OCR", "1") == "1"
OCR_MIN_CHARS = 40  # a page with fewer extracted chars is treated as scanned

# ---------------- Scoring ----------------
RISK_LABELS = ["Strong Coverage", "Moderate Coverage", "Caution Required", "High Risk"]
LABEL_DISPLAY = {"High Risk": "High Risk / Needs Advisor Review"}

MODEL_FEATURES = [
    "sum_insured", "annual_premium", "premium_to_coverage_ratio",
    "waiting_period_months", "pre_existing_disease_waiting_period_months",
    "copayment_percentage", "room_rent_limit_present", "icu_limit_present",
    "disease_sub_limit_count", "exclusion_count", "restoration_benefit_present",
    "no_claim_bonus_present", "child_coverage_present", "maternity_coverage_present",
    "claim_process_clarity_score", "ambiguous_clause_count", "risk_flags_count",
]
CATEGORICAL_FEATURES = ["policy_type"]
POLICY_TYPES = ["Family Floater", "Individual Health", "Senior Citizen",
                "Group Health", "Top-up Health", "Critical Illness"]

DISCLAIMER = (
    "CoverWise AI is a policy-understanding and decision-support assistant. It does NOT "
    "provide financial, legal, medical or insurance advice and does not recommend buying, "
    "switching or cancelling any policy. AI extraction can be incomplete or wrong. Always "
    "verify every point with the official policy wording, the insurer, or a licensed "
    "insurance advisor before making a decision."
)