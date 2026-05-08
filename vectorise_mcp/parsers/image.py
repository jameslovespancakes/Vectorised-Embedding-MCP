"""Image parser. Pure-OCR — yields recognized text or nothing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from vectorise_mcp import ocr

logger = logging.getLogger(__name__)


def parse(path: Path) -> Iterator[tuple[str, None]]:
    if not ocr.is_available():
        logger.warning(
            "Skipping image %s — OCR deps not installed. "
            "Install with: pip install vectorise-mcp[ocr]",
            path,
        )
        return
    text = ocr.ocr_image(str(path)).strip()
    if text:
        yield text, None
