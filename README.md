# vectorise-mcp

Local MCP server that turns folders of documents into a hybrid vector + keyword index that Claude Desktop can search. Stays offline after first model download.

[![PyPI](https://img.shields.io/pypi/v/vectorise-mcp)](https://pypi.org/project/vectorise-mcp/)

## Stack

- **MCP**: [`mcp`](https://github.com/modelcontextprotocol/python-sdk) (FastMCP), stdio transport
- **Embeddings**: [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) (384-dim)
- **Reranker**: [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) cross-encoder
- **Vector DB**: [`sqlite-vec`](https://github.com/asg017/sqlite-vec)
- **Keyword DB**: SQLite FTS5 (BM25)
- **Fusion**: Reciprocal Rank Fusion → cross-encoder rerank

## Install

```bash
pip install vectorise-mcp                 # core
pip install "vectorise-mcp[ocr]"          # + OCR for scanned PDFs / images
pip install "vectorise-mcp[notify]"       # + desktop toast on job completion
pip install "vectorise-mcp[ocr,notify]"   # everything

vectorise-mcp setup                       # pre-download models (~250MB)
```

Python ≥ 3.10.

## Wire into Claude Desktop

`claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "vectorise": {
      "command": "vectorise-mcp",
      "args": ["serve"]
    }
  }
}
```

Config file location:
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

Restart Claude Desktop.

## File support

| Format | Notes |
|---|---|
| `.pdf` | text + OCR fallback for scanned pages |
| `.docx`, `.pptx`, `.xlsx`, `.xlsm`, `.xls` | full content + tables |
| `.txt`, `.md`, `.markdown` | UTF-8 |
| `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp` | OCR (requires `[ocr]`) |
| `.doc`, `.ppt` | detected, skipped, reported |

## Tools exposed to Claude

| Tool | What it does |
|---|---|
| `vectorise_list_projects` | list all indexed projects |
| `vectorise_index_project(folder, project, mode)` | start indexing job, returns `job_id` instantly |
| `vectorise_reindex_project(project)` | SHA1-incremental rescan of all sources |
| `vectorise_index_status(job_id)` | instant job snapshot incl. progress + ETA |
| `vectorise_await_index(job_id, timeout_sec)` | optional blocking wait |
| `vectorise_list_jobs(active_only)` | jobs from current server session |
| `vectorise_search(project, query, k, candidate_pool, file_glob, subdirectory, page_min, page_max, min_similarity)` | hybrid + reranked search |
| `vectorise_delete_project(project)` | delete project's `.db` |

`mode` for `vectorise_index_project`: `auto` (default — incremental if path already indexed, error on conflict) / `replace` / `append` / `fail`.

## Architecture

Indexing job runs in a daemon thread with its own asyncio loop. The MCP server's main loop stays free to serve `index_status` / `search` calls regardless of how heavy the embedding/OCR work is. Status calls are instant; search works on the partial index while a job is running.

```
folder
  ↓  parsers.parse                        (.pdf .docx .pptx .xlsx ...)
chunks (sentence-aware, 384 tok / 96 overlap, single-sentence hard-split)
  ↓  embedder.embed_passages              (BGE-small)
sqlite-vec   +   FTS5 (BM25)              ← per-file SHA1 dedup, basename collision auto-rename
  ↓  search                               (vector top-N + BM25 top-N)
RRF fusion → cross-encoder rerank → top-K
```

Project DBs live in `~/.vectorise-mcp/<name>.db`. Self-contained — source folder can be deleted after indexing.

## Config (env vars)

| Var | Default | Purpose |
|---|---|---|
| `VECTORISE_MCP_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | must be 384-dim |
| `VECTORISE_MCP_RERANKER_MODEL` | `BAAI/bge-reranker-base` | |
| `VECTORISE_MCP_EMBED_BATCH` | `32` | |
| `VECTORISE_MCP_RERANKER_BATCH` | `16` | |
| `VECTORISE_MCP_OCR_MIN_CONFIDENCE` | `0.5` | drop OCR lines below |
| `VECTORISE_MCP_OCR_WORKERS` | `4` | parallel page OCR threads |
| `VECTORISE_MCP_OCR_DPI` | `200` | PDF rasterisation DPI |
| `VECTORISE_MCP_OCR_MAX_DIM` | `4000` | downscale huge images before OCR |
| `VECTORISE_MCP_NOTIFY` | `1` | desktop toast on/off |

## Performance

| | CPU | GPU |
|---|---|---|
| Indexing throughput | ~80 chunks/sec | 5–10× faster |
| Search latency (k=5, ≤500K chunks) | ~150ms | similar |
| Disk per chunk | ~2 KB | |
| Cold start | ~5s (lazy model load) | |

## Local dev

```bash
git clone https://github.com/jameslovespancakes/Vectorised-Embedding-MCP
cd Vectorised-Embedding-MCP
pip install -e ".[ocr,notify]"

# tests bypass MCP transport, drive indexer + tools directly
python tests/smoke_test.py
python tests/smoke_test_projects.py
python tests/smoke_test_jobs.py
python tests/smoke_test_filters.py
python tests/smoke_test_office.py
python tests/smoke_test_chunking.py
python tests/smoke_test_legacy_skip.py
```

## License

MIT.
