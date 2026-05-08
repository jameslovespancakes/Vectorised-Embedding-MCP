"""Optional OCR for scanned PDFs and image files.

Engine:    rapidocr-onnxruntime  (pure Python, ONNX, no system Tesseract install)
Rasterize: pypdfium2             (pure Python, no Poppler)

Both deps optional: `pip install vectorise-mcp[ocr]`. When unavailable, parsers
fall back to text-only behavior.

Quality controls:
  • Confidence filtering — drop OCR lines below VECTORISE_MCP_OCR_MIN_CONFIDENCE
    (default 0.5). Removes watermark/page-edge noise that hurts retrieval.
  • Parallel page OCR — when several PDF pages need OCR, render sequentially
    (pypdfium2 is not thread-safe per document) then OCR in a thread pool.
    ONNX releases the GIL during inference → ~4-8x speedup on multi-page scans.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE = float(os.environ.get("VECTORISE_MCP_OCR_MIN_CONFIDENCE", "0.5"))
DEFAULT_MAX_WORKERS = int(os.environ.get("VECTORISE_MCP_OCR_WORKERS", "4"))
DEFAULT_DPI = int(os.environ.get("VECTORISE_MCP_OCR_DPI", "200"))
MAX_IMAGE_DIM = int(os.environ.get("VECTORISE_MCP_OCR_MAX_DIM", "4000"))

_ocr_engine = None
_ocr_lock = threading.Lock()
_availability_checked = False
_available = False


def is_available() -> bool:
    """True if OCR deps installed and importable."""
    global _availability_checked, _available
    if _availability_checked:
        return _available
    try:
        import rapidocr_onnxruntime  # noqa: F401
        import pypdfium2  # noqa: F401
        _available = True
    except ImportError:
        _available = False
    _availability_checked = True
    return _available


def warm_up() -> None:
    """Force OCR engine init (downloads ~30MB ONNX models on first run)."""
    if is_available():
        _get_engine()


def _get_engine():
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine
    with _ocr_lock:
        if _ocr_engine is None:
            from rapidocr_onnxruntime import RapidOCR
            logger.info("Loading RapidOCR engine (one-time download if absent)")
            _ocr_engine = RapidOCR()
    return _ocr_engine


def _filter_lines(result, min_confidence: float) -> str:
    """Extract text from RapidOCR result, keeping only lines with confidence >= threshold.

    RapidOCR returns list of [bbox, text, confidence] entries (or None).
    """
    if not result:
        return ""
    kept: list[str] = []
    for entry in result:
        if not entry or len(entry) < 3:
            continue
        text = entry[1]
        try:
            conf = float(entry[2])
        except (TypeError, ValueError):
            conf = 0.0
        if text and conf >= min_confidence:
            kept.append(text)
    return "\n".join(kept)


def ocr_image(
    image_path_or_bytes,
    min_confidence: float = DEFAULT_CONFIDENCE,
) -> str:
    """OCR a single image (path str, bytes, or PIL.Image-compatible). Returns concatenated
    text from lines whose confidence >= `min_confidence`.
    """
    if not is_available():
        return ""
    engine = _get_engine()
    result, _elapsed = engine(image_path_or_bytes)
    return _filter_lines(result, min_confidence)


def _render_pdf_page(pdf_path: Path, page_index: int, dpi: int) -> bytes | None:
    """Render one PDF page to PNG bytes. Caller is responsible for serializing renders."""
    try:
        import pypdfium2 as pdfium
        from PIL import Image
    except ImportError:
        return None
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[page_index]
        scale = dpi / 72.0
        pil_image = page.render(scale=scale).to_pil()
        # Normalize: clamp huge dimensions, force RGB.
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        if max(pil_image.size) > MAX_IMAGE_DIM:
            pil_image.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.LANCZOS)
        buf = BytesIO()
        pil_image.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        logger.exception("render failed for %s page %d", pdf_path, page_index)
        return None
    finally:
        pdf.close()


def ocr_pdf_page(
    pdf_path: Path,
    page_index: int,
    dpi: int = DEFAULT_DPI,
    min_confidence: float = DEFAULT_CONFIDENCE,
) -> str:
    """Rasterize one PDF page (0-indexed) and OCR it."""
    if not is_available():
        return ""
    img_bytes = _render_pdf_page(pdf_path, page_index, dpi)
    if not img_bytes:
        return ""
    return ocr_image(img_bytes, min_confidence=min_confidence)


def ocr_pdf_pages(
    pdf_path: Path,
    page_indices: list[int],
    dpi: int = DEFAULT_DPI,
    min_confidence: float = DEFAULT_CONFIDENCE,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[int, str]:
    """OCR multiple PDF pages with parallel inference. Returns {page_index: text}.

    Rendering happens serially (pypdfium2 not thread-safe per document); OCR runs
    in a thread pool — ONNX releases the GIL during inference.
    """
    if not is_available() or not page_indices:
        return {idx: "" for idx in page_indices}

    rendered: dict[int, bytes | None] = {}
    for idx in page_indices:
        rendered[idx] = _render_pdf_page(pdf_path, idx, dpi)

    def _ocr_one(idx: int) -> tuple[int, str]:
        img = rendered.get(idx)
        if not img:
            return idx, ""
        return idx, ocr_image(img, min_confidence=min_confidence)

    workers = max(1, min(max_workers, len(page_indices)))
    results: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for idx, text in ex.map(_ocr_one, page_indices):
            results[idx] = text
    return results
