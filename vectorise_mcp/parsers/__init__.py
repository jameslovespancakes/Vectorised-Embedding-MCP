"""Parser dispatch by file extension. Each parser yields (text, page_or_none) tuples."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from vectorise_mcp.parsers import docx as docx_parser
from vectorise_mcp.parsers import image as image_parser
from vectorise_mcp.parsers import pdf as pdf_parser
from vectorise_mcp.parsers import pptx as pptx_parser
from vectorise_mcp.parsers import spreadsheet as spreadsheet_parser
from vectorise_mcp.parsers import text as text_parser

TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
SPREADSHEET_EXTENSIONS = {".xlsx", ".xlsm", ".xls"}
PRESENTATION_EXTENSIONS = {".pptx"}
WORD_EXTENSIONS = {".docx"}

# Detected during scan but NOT supported. Indexer reports + skips these so
# Claude can tell the user, rather than silently dropping them.
UNSUPPORTED_LEGACY_EXTENSIONS = {".doc", ".ppt"}

SUPPORTED_EXTENSIONS = (
    {".pdf"}
    | TEXT_EXTENSIONS
    | IMAGE_EXTENSIONS
    | WORD_EXTENSIONS
    | PRESENTATION_EXTENSIONS
    | SPREADSHEET_EXTENSIONS
)


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def is_unsupported_legacy(path: Path) -> bool:
    return path.suffix.lower() in UNSUPPORTED_LEGACY_EXTENSIONS


def parse(path: Path) -> Iterator[tuple[str, int | None]]:
    """Dispatch by extension. Yields (text_block, page_or_none)."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        yield from pdf_parser.parse(path)
    elif ext in WORD_EXTENSIONS:
        yield from docx_parser.parse(path)
    elif ext in PRESENTATION_EXTENSIONS:
        yield from pptx_parser.parse(path)
    elif ext in SPREADSHEET_EXTENSIONS:
        yield from spreadsheet_parser.parse(path)
    elif ext in TEXT_EXTENSIONS:
        yield from text_parser.parse(path)
    elif ext in IMAGE_EXTENSIONS:
        yield from image_parser.parse(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")
