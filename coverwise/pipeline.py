"""End-to-end orchestration used by the Streamlit app, the CLI and the evaluation scripts."""
from __future__ import annotations

import time

from .analysis import advisor_questions, compare_policies, generate_risk_flags
from .extraction import extract_clauses
from .features import build_features
from .llm import LLMClient
from .pdf_extract import extract_policy_text
from .qa import policy_summary
from .rag import build_rag_index, chunk_sections
from .scoring import score_policy
from .text_clean import clean_pages, split_sections


def analyze_policy(src, policy_name: str, llm: LLMClient | None = None, profile: dict | None = None,
                   overrides: dict | None = None, progress=None) -> dict:
    """Run the full single-policy pipeline. `progress(fraction, message)` is optional."""
    t0 = time.time()
    step = progress or (lambda *_: None)
    overrides = {k: v for k, v in (overrides or {}).items() if v}

    step(0.05, f"{policy_name}: extracting text")
    raw_pages = extract_policy_text(src)
    pages = clean_pages(raw_pages)
    if sum(len(p["text"]) for p in pages) < 200:
        raise ValueError(f"{policy_name}: almost no text could be extracted (scanned PDF without OCR?).")

    step(0.2, f"{policy_name}: sectioning & indexing")
    blocks = split_sections(pages)
    chunks = chunk_sections(blocks, policy_name)
    index = build_rag_index(chunks)

    step(0.4, f"{policy_name}: extracting clauses")
    extraction = extract_clauses(blocks, pages, index, llm, raw_pages)

    step(0.7, f"{policy_name}: scoring & risk flags")
    features, unclear = build_features(extraction, overrides)
    score = score_policy(features)
    flags = generate_risk_flags(extraction, features, unclear, profile)

    analysis = {
        "name": policy_name, "pages": len(pages), "ocr_pages": sum(p["ocr"] for p in raw_pages),
        "chunks": chunks, "index": index, "blocks": blocks, "extraction": extraction,
        "features": features, "unclear": unclear, "score": score, "flags": flags, "overrides": overrides,
    }
    step(0.85, f"{policy_name}: writing summary")
    analysis["summary"] = policy_summary(extraction, features, score, index, llm)
    analysis["advisor_questions"] = advisor_questions(analysis, None, profile, llm)
    analysis["processing_seconds"] = round(time.time() - t0, 1)
    step(1.0, f"{policy_name}: done")
    return analysis


def analyze_pair(current_src, proposed_src=None, llm: LLMClient | None = None, profile: dict | None = None,
                 current_overrides: dict | None = None, proposed_overrides: dict | None = None,
                 progress=None) -> dict:
    current = analyze_policy(current_src, "Current policy", llm, profile, current_overrides, progress)
    result = {"current": current, "proposed": None, "comparison": None}
    if proposed_src is not None:
        proposed = analyze_policy(proposed_src, "Proposed policy", llm, profile, proposed_overrides, progress)
        comparison = compare_policies(current, proposed, profile, llm)
        proposed["advisor_questions"] = advisor_questions(proposed, comparison, profile, llm)
        result.update(proposed=proposed, comparison=comparison)
    return result
