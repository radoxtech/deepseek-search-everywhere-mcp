# DeepSeek Search Everywhere MCP

A research-grade MCP server for the official DeepSeek Search & Fetch API.

> **Status:** v0.2 in development. The implementation plan is in [`docs/plan-v0.2.md`](docs/plan-v0.2.md).

Not affiliated with DeepSeek.

## Why another DeepSeek MCP?

Most wrappers pass JSON through. This one adds what repeated research actually needs:

- **Canonical-URL deduplication** with a guarded title/site rule (available now)
- **Optional query rewrite** via `deepseek-flash` with thinking disabled (available now)
- **Per-result and total character budgets** (available now)
- **Response cache** with TTL, LRU and single-flight (planned v0.2)
- **Proactive per-endpoint rate limiting** (planned v0.2)
- **Batch fetch and research composition** (planned v0.2)

## Tools (current baseline)

| Tool | Purpose | Key parameters |
|---|---|---|
| `deepseek_web_search` | Search the web and return titles, URLs, dates, authority and rerank scores | `query`, `count=10`, `need_content=False`, `ori_question`, `lang`, `max_chars_per_result=4000`, `max_total_chars=40000`, `dedupe=True`, `rewrite=False` |
| `deepseek_search_and_read` | Search and return page content in one request | `query`, `count=5`, `ori_question`, `lang`, `max_chars_per_result=4000`, `max_total_chars=30000`, `dedupe=True`, `rewrite=False` |
| `deepseek_fetch` | Fetch one known URL and return its Markdown content | `url`, `clean=True`, `max_chars=20000` |
| `deepseek_balance` | Check the DeepSeek account balance | — |

`deepseek_fetch` returns HTTP 404 for PDFs and Reddit, and times out on some sites. For those, use `deepseek_search_and_read` with the page title or a `site:`-scoped query.

## Quick start (current baseline)

Requirements: Python 3.11+ and a DeepSeek API key.

```bash
git clone https://github.com/radoxtech/deepseek-search-everywhere-mcp.git
cd deepseek-search-everywhere-mcp

python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[test]"

export DEEPSEEK_API_KEY=sk-...
pytest -q
python server.py --check   # optional: verifies the key against the balance endpoint
```

MCP client configuration (stdio):

```json
{
  "mcpServers": {
    "deepseek-search-everywhere": {
      "command": "python",
      "args": ["/absolute/path/to/deepseek-search-everywhere-mcp/server.py"],
      "env": {
        "DEEPSEEK_API_KEY": "sk-..."
      }
    }
  }
}
```

After the v0.2 packaging work, the intended installation path is:

```bash
uvx deepseek-search-everywhere-mcp
```

## Configuration

| Variable | Required | Purpose |
|---|---|---|
| `DEEPSEEK_API_KEY` | Yes | DeepSeek platform key |
| `DEEPSEEK_BASE_URL` | No | Override the API base URL (default `https://api.deepseek.com`) |

Legacy convenience: if `DEEPSEEK_API_KEY` is not set, the server can read an opencode credential file at `~/.local/share/opencode/auth.json`. This is intentionally documented as an opencode-specific fallback, not the default path.

## Development

```bash
pytest -q          # offline tests; no network calls
```

The v0.2 plan adds `ruff`, a `src/` layout, a console entry point, a CI matrix and packaging.

## Roadmap

The full plan — 29 tasks, 6 waves + Wave 0, feature scoring and acceptance criteria — lives in [`docs/plan-v0.2.md`](docs/plan-v0.2.md).

## License

MIT. The `LICENSE` file will be added during the v0.2 packaging work.
