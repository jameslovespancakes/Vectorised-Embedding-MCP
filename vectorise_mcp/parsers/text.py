"""Plain text / markdown parser. Reads file as UTF-8."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator


def parse(path: Path) -> Iterator[tuple[str, None]]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if text:
        yield text, None
