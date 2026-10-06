"""Step 1 - Insurance policy PDF ingestion and text extraction.

* Digital PDFs  -> PyMuPDF (fast, keeps reading order) with pdfplumber fallback.
* Scanned pages -> OCR with Tesseract (only for pages with almost no text).
Returns a list of {"page": int, "text": str, "ocr": bool}.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import BinaryIO, Union

from . import config

log = logging.getLogger(__name__)

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24
except ImportError:  # pragma: no cover
    try:
        import fitz  # older PyMuPDF
    except ImportError:
        fitz = None

PdfInput = Union[str, Path, bytes, BinaryIO]


def _read_bytes(src: PdfInput) -> bytes:
    if isinstance(src, (str, Path)):
        return Path(src).read_bytes()
    if isinstance(src, bytes):
        return src
    if hasattr(src, "getvalue"):          # Streamlit UploadedFile / BytesIO
        return src.getvalue()
    return src.read()


def _ocr_page(page) -> str:
    """OCR one PyMuPDF page. Needs the tesseract binary (apt install tesseract-ocr)."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        log.warning("pytesseract/Pillow not installed - skipping OCR")
        return ""
    try:
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img, lang="eng")
    except Exception as exc:  # tesseract binary missing etc.
        log.warning("OCR failed on page %s: %s", page.number + 1, exc)
        return ""


def _extract_with_pymupdf(data: bytes, enable_ocr: bool) -> list[dict]:
    pages = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            # content-stream order keeps table cells together (CIS tables); sort=True interleaves columns
            text = page.get_text("text") or ""
            used_ocr = False
            if enable_ocr and len(text.strip()) < config.OCR_MIN_CHARS:
                ocr_text = _ocr_page(page)
                if len(ocr_text.strip()) > len(text.strip()):
                    text, used_ocr = ocr_text, True
            pages.append({"page": page.number + 1, "text": text, "ocr": used_ocr})
    return pages


def _extract_with_pdfplumber(data: bytes) -> list[dict]:
    import pdfplumber
    pages = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            pages.append({"page": i, "text": page.extract_text() or "", "ocr": False})
    return pages


def extract_policy_text(src: PdfInput, enable_ocr: bool | None = None) -> list[dict]:
    """Extract page-wise text from a policy PDF (path, bytes or uploaded file)."""
    enable_ocr = config.ENABLE_OCR if enable_ocr is None else enable_ocr
    data = _read_bytes(src)
    if not data[:5] == b"%PDF-":
        raise ValueError("Uploaded file is not a valid PDF.")
    if fitz is not None:
        try:
            return _extract_with_pymupdf(data, enable_ocr)
        except Exception as exc:
            log.warning("PyMuPDF failed (%s); falling back to pdfplumber", exc)
    return _extract_with_pdfplumber(data)
