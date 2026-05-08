"""PDF parser via pypdf. Per-page extraction. Auto-falls-back to OCR for pages
whose text layer is empty or near-empty (likely scanned), if OCR deps installed.

Multi-page OCR is batched: all scan-suspected pages in a single PDF go through
`ocr.ocr_pdf_pages` together, which renders serially then OCRs in a thread pool.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader

from vectorise_mcp import ocr

logger = logging.getLogger(__name__)

# A page with fewer than this many alphabetic chars is treated as scanned/empty.
_OCR_FALLBACK_THRESHOLD = 50
_ALPHA_RE = re.compile(r"[A-Za-z]")


def _alpha_count(s: str) -> int:
    return len(_ALPHA_RE.findall(s))


def parse(path: Path) -> Iterator[tuple[str, int]]:
    reader = PdfReader(str(path))
    ocr_avail = ocr.is_available()

    # Pass 1: extract text-layer text; flag pages that look scanned.
    page_texts: list[str] = []
    needs_ocr: list[int] = []  # 0-indexed
    for page_idx, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if _alpha_count(text) < _OCR_FALLBACK_THRESHOLD and ocr_avail:
            needs_ocr.append(page_idx)
        page_texts.append(text)

    # Pass 2: batch-OCR all suspected scanned pages in parallel.
    if needs_ocr:
        logger.info("OCR fallback for %s — %d page(s) flagged as scanned",
                    path.name, len(needs_ocr))
        ocr_results = ocr.ocr_pdf_pages(path, needs_ocr)
        for idx in needs_ocr:
            ocr_text = (ocr_results.get(idx) or "").strip()
            if _alpha_count(ocr_text) > _alpha_count(page_texts[idx]):
                page_texts[idx] = ocr_text

    for page_idx, text in enumerate(page_texts):
        if text:
            yield text, page_idx + 1
