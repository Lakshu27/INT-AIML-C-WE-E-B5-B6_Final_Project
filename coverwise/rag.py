"""Chunking + retrieval index for RAG-based policy Q&A.

Hybrid retrieval:
  * dense  : sentence-transformers embeddings + FAISS (cosine)
  * sparse : TF-IDF (word 1-2 grams) - great for exact insurance terms like
             "co-payment", "Excl02", "room rent"
If sentence-transformers / FAISS are not installed (or the model cannot be
downloaded), the index silently runs in TF-IDF-only mode.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config

log = logging.getLogger(__name__)
_EMBEDDER = None  # module-level cache so the model loads once


@dataclass
class Chunk:
    chunk_id: str
    policy_name: str
    text: str
    page_start: int
    page_end: int
    section_title: str
    section_category: str

    @property
    def citation(self) -> str:
        pages = f"p.{self.page_start}" if self.page_start == self.page_end \
            else f"pp.{self.page_start}-{self.page_end}"
        return f"{self.policy_name}, {pages}, \u00a7 {self.section_title}"

    def to_dict(self) -> dict:
        return {**self.__dict__, "citation": self.citation}


def chunk_sections(blocks: list[dict], policy_name: str,
                   size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[Chunk]:
    """Split section blocks into overlapping chunks that never cross a section."""
    chunks: list[Chunk] = []
    n = 0
    for block in blocks:
        lines = block["lines"]
        buf: list[tuple[int, str]] = []
        length = 0
        i = 0
        while i < len(lines):
            page, text = lines[i]
            buf.append((page, text))
            length += len(text) + 1
            i += 1
            if length >= size or i == len(lines):
                body = "\n".join(t for _, t in buf)
                if len(body.strip()) > 25:
                    n += 1
                    chunks.append(Chunk(
                        chunk_id=f"{policy_name[:12]}-{n:04d}",
                        policy_name=policy_name,
                        text=body if body.startswith(block["section_title"])
                        else f"[{block['section_title']}]\n{body}",
                        page_start=buf[0][0], page_end=buf[-1][0],
                        section_title=block["section_title"],
                        section_category=block["section_category"],
                    ))
                # keep an overlap tail
                tail, tail_len = [], 0
                for item in reversed(buf):
                    if tail_len >= overlap or i == len(lines):
                        break
                    tail.insert(0, item)
                    tail_len += len(item[1])
                buf, length = tail, tail_len
    return chunks


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is not None or not config.USE_EMBEDDINGS:
        return _EMBEDDER or None
    try:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(config.EMBEDDING_MODEL)
    except Exception as exc:
        log.warning("Embeddings unavailable (%s) - using TF-IDF only", exc)
        _EMBEDDER = False
    return _EMBEDDER or None


@dataclass
class PolicyIndex:
    chunks: list[Chunk]
    tfidf: TfidfVectorizer = field(init=False)
    tfidf_matrix: object = field(init=False)
    faiss_index: object = field(init=False, default=None)
    embedder: object = field(init=False, default=None)

    def __post_init__(self):
        texts = [c.text for c in self.chunks] or [""]
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                     stop_words="english", min_df=1)
        self.tfidf_matrix = self.tfidf.fit_transform(texts)
        self.embedder = _get_embedder()
        if self.embedder is not None and self.chunks:
            try:
                import faiss
                vecs = self.embedder.encode(texts, normalize_embeddings=True,
                                            show_progress_bar=False).astype("float32")
                self.faiss_index = faiss.IndexFlatIP(vecs.shape[1])
                self.faiss_index.add(vecs)
            except Exception as exc:
                log.warning("FAISS unavailable (%s) - TF-IDF only", exc)
                self.faiss_index = None

    @property
    def mode(self) -> str:
        return "hybrid (FAISS + TF-IDF)" if self.faiss_index is not None else "TF-IDF"

    def search(self, query: str, k: int = config.TOP_K,
               categories: list[str] | None = None) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        q = _expand_query(query)
        sparse = (self.tfidf.transform([q]) @ self.tfidf_matrix.T).toarray().ravel()
        scores = sparse
        if self.faiss_index is not None:
            qv = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
            dense_scores, idx = self.faiss_index.search(qv, len(self.chunks))
            dense = np.zeros(len(self.chunks))
            dense[idx[0]] = dense_scores[0]
            scores = 0.45 * sparse + 0.55 * np.clip(dense, 0, None)
        if not re.search(r"\bmean|defin", query, re.I):  # prefer operative clauses over definitions
            scores = scores * np.array([0.8 if c.section_category == "definitions" else 1.0
                                        for c in self.chunks])
        if categories:
            boost = np.array([1.15 if c.section_category in categories else 1.0
                              for c in self.chunks])
            scores = scores * boost
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]


# insurance synonyms so lay questions hit legal wording
_SYNONYMS = {
    "copay": "co-payment copayment co-pay cost sharing",
    "co-pay": "co-payment copayment cost sharing",
    "room": "room rent boarding nursing single private room",
    "icu": "intensive care unit iccu",
    "pre-existing": "pre-existing disease ped excl01 waiting period",
    "ped": "pre-existing disease excl01",
    "waiting": "waiting period months continuous coverage excl01 excl02 excl03",
    "child": "dependent child children newborn baby age",
    "newborn": "new born baby maternity",
    "baby": "new born newborn maternity",
    "maternity": "maternity delivery pregnancy childbirth newborn",
    "restoration": "restoration restore restored recharge refill",
    "restore": "restoration restored recharge",
    "exclusion": "exclusions excluded not covered excl",
    "claim": "claim cashless reimbursement intimation documents tpa",
    "switch": "portability migration continuity waiting period credit",
    "ncb": "no claim bonus cumulative bonus",
    "bonus": "no claim bonus cumulative bonus",
    "renew": "renewal grace period lifelong",
}


def _expand_query(q: str) -> str:
    low = q.lower()
    extra = [v for k, v in _SYNONYMS.items() if re.search(rf"\b{re.escape(k)}", low)]
    return q + " " + " ".join(extra)


def build_rag_index(chunks: list[Chunk]) -> PolicyIndex:
    return PolicyIndex(chunks)
