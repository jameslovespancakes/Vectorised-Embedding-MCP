"""CLI entry point.

  vectorise-mcp setup    # download embedder + reranker into HF cache
  vectorise-mcp serve    # run stdio MCP server (use this in claude_desktop_config.json)
  vectorise-mcp version  # print version
"""

from __future__ import annotations

import argparse
import sys

from vectorise_mcp import __version__


def _cmd_serve(_args) -> int:
    from vectorise_mcp.server import run_stdio
    run_stdio()
    return 0


def _cmd_setup(_args) -> int:
    """Download embedder + reranker (+ OCR if available) so subsequent boots are instant + offline."""
    from vectorise_mcp import embedder, ocr, reranker
    print(f"[setup] downloading embedder: {embedder.DEFAULT_MODEL}", flush=True)
    embedder.warm_up()
    print(f"[setup] downloading reranker: {reranker.DEFAULT_MODEL}", flush=True)
    reranker.warm_up()
    if ocr.is_available():
        print("[setup] warming OCR engine (RapidOCR)", flush=True)
        ocr.warm_up()
    else:
        print("[setup] OCR deps not installed (skipped). For scanned PDFs / images:")
        print("        pip install vectorise-mcp[ocr]")
    print("[setup] done. Models cached. Ready for `vectorise-mcp serve`.")
    return 0


def _cmd_version(_args) -> int:
    print(__version__)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="vectorise-mcp",
        description="Local MCP server: vectorise folders into a hybrid embedding DB for Claude.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="Run stdio MCP server (for Claude Desktop)").set_defaults(
        func=_cmd_serve
    )
    sub.add_parser(
        "setup",
        help="Pre-download embedder + reranker models (~250MB total, one-time)",
    ).set_defaults(func=_cmd_setup)
    sub.add_parser("version", help="Print version").set_defaults(func=_cmd_version)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
