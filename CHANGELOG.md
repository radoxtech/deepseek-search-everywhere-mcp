# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-27

### Added

- Package layout under `src/deepseek_search_everywhere_mcp` with a `deepseek-search-everywhere-mcp` console script
  (`--version`, transport options) and a top-level `server.py` shim for existing configurations.
- In-memory TTL + LRU response cache with a byte budget, single-flight for identical concurrent calls and a
  `refresh` parameter to bypass it.
- Per-endpoint token-bucket rate limiter with a maximum wait (`rate_limited_local`).
- Response size caps on streamed upstream bodies, and `Retry-After` date support.
- Secret redaction in error messages and notes.
- Static URL and SSRF validation before the cache, the limiter or the network are touched.
- `deepseek_fetch_many` tool for batch fetching with a shared deadline and bounded concurrency; `partial` and
  `failed_count` flag per-URL failures (`deepseek_research` reports `failed_count` too).
- `deepseek_research` tool that searches and fills gaps with follow-up fetches; each source carries `date` and `site`, and `lang`,
  `rewrite` and `dedupe` work as in search (`query_used` reports the query sent).
- Streamable HTTP transport with bearer-token authentication (`http` extra).
- Search over-fetches upstream so `dedupe` still yields the requested number of results.
- Blank result titles fall back to the URL host; a fetched page title replaces the fallback in research results.
- `server.json` for the MCP Registry and a structural validator, `scripts/validate_server_json.py`.
- CI workflow (Python 3.11-3.13, lint, tests, build, `twine check`) and a tag-triggered release workflow
  (PyPI Trusted Publishing, GitHub Release, MCP Registry).
- LLM-judged end-to-end live tests (opt-in with `-m live`).
- PyPI metadata: keywords, classifiers, Changelog and Documentation URLs, and a `py.typed` marker.

### Changed

- `clean=True` removes backslash escapes from all ASCII punctuation (`\(`, `\-`, `\[`, `\/`...), not just a few,
  and leaves fenced code blocks and inline code spans untouched.
- The query-rewrite prompt and user-facing notes are language-neutral.
- Notes are human-readable text in every tool; content cut to a character budget is reported with the same
  sentence by search, `deepseek_fetch_many` and `deepseek_research` (no `truncated_to_budget` token).
- `cached` is a boolean per response and per item; batch tools report the number of hits as `cached_count`.
- `--check` no longer prints the length of the API key.
- Settings are a frozen dataclass with documented defaults; `batch_concurrency` is capped at 5.
- `DEEPSEEK_MCP_LOG_LEVEL` sets the level of the server's, the MCP SDK's and uvicorn's logs and defaults to
  `WARNING`, so the SDK no longer logs every request at INFO.
- The 429 hint no longer depends on upstream rate limits.
- Retry backoff uses an injectable sleep function.
- The transport's `retries` argument is renamed `attempts`: it always counted total tries, so the default of 2
  means one retry.
- The license is declared as a PEP 639 SPDX expression.

### Security

- Secrets are redacted from errors and notes.
- Internal hosts, literal private or loopback addresses and URLs with embedded credentials are rejected by static
  validation (no local DNS resolution).
- Upstream bodies are size-capped while streaming.
- The HTTP transport requires a bearer token.

## [0.1.0]

Initial baseline: a single-file `server.py` MCP server exposing DeepSeek web search and fetch tools over stdio.

[0.2.0]: https://github.com/radoxtech/deepseek-search-everywhere-mcp/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/radoxtech/deepseek-search-everywhere-mcp/releases/tag/v0.1.0
