"""Optional cross-platform desktop notifications when long-running jobs complete.

MCP servers cannot wake Claude up unsolicited — Claude only runs on user input
or tool-call returns. Best we can do for the "ping when done" UX is fire a
native OS toast that the user sees in their notification center / system tray.
The user then knows to return to the Claude Desktop conversation and ask Claude
for the result.

Backend: `plyer` (pure Python, cross-platform — Windows toast, macOS Notification
Center, Linux libnotify). Optional dep: `pip install vectorise-mcp[notify]`.

Failures are silent — notification is a nicety, not a correctness mechanism.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

# Master switch via env var; default on if backend importable.
ENABLED = os.environ.get("VECTORISE_MCP_NOTIFY", "1") not in ("0", "false", "False", "")

_availability_checked = False
_available = False
_check_lock = threading.Lock()


def is_available() -> bool:
    global _availability_checked, _available
    if _availability_checked:
        return _available and ENABLED
    with _check_lock:
        if _availability_checked:
            return _available and ENABLED
        try:
            from plyer import notification  # noqa: F401
            _available = True
        except ImportError:
            _available = False
        _availability_checked = True
    return _available and ENABLED


def notify(title: str, message: str, timeout: int = 10) -> None:
    """Fire a desktop toast. Silent on failure / when deps missing / when disabled."""
    if not is_available():
        return
    try:
        from plyer import notification
        if notification is None:
            return
        notification.notify(
            title=title[:60],
            message=message[:240],
            app_name="vectorise-mcp",
            timeout=timeout,
        )
    except Exception:
        logger.debug("desktop notification failed", exc_info=True)
