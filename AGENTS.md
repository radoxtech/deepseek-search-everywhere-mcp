# AGENTS.md

## Project overview

`deepseek-search-everywhere-mcp` is a Python MCP server exposing web search, page fetch, batch fetch and
research tools over the DeepSeek Search & Fetch API (stdio by default, optional bearer-protected HTTP).

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test,lint,http]" build twine
# uv equivalent: uv venv && uv pip install -e ".[test,lint,http]" build twine
```

## Commands

```bash
pytest -q                                   # offline tests (default; no network, no key)
cp .env.example .env                        # then add your own DEEPSEEK_API_KEY
set -a; . ./.env; set +a; pytest -m live    # live tests against the real API (opt-in)
ruff check . && ruff format --check .       # lint and format check
python -m build && twine check dist/*       # build and validate distributions
python scripts/validate_server_json.py      # validate server.json
scripts/smoke_package.sh                    # install the built wheel in a clean venv and run the CLI
```

## Code layout (`src/deepseek_search_everywhere_mcp/`)

- `server.py`: FastMCP server; tool definitions, result shaping, query rewrite.
- `transport.py`: HTTP calls to the DeepSeek API; credentials, retries, bounded response reads.
- `cache.py`: TTL + LRU response cache with single-flight.
- `ratelimit.py`: per-endpoint token-bucket limiter.
- `settings.py`: defaults as constants and the two environment switches.
- `urls.py`: static URL validation (SSRF abuse filter) for fetch tools.
- `errors.py`: error envelopes and stable error codes.
- `redact.py`: secret redaction for error messages and notes.
- `boilerplate.py`: removes known site-UI boilerplate from page content.
- `http_app.py`: streamable HTTP app with mandatory bearer token.
- `cli.py`, `__main__.py`: command-line entry point (`--check`, transport selection).
- `_version.py`: single source of the installed version.

Tests live in `tests/` (`tests/live/` holds the opt-in live suite); `scripts/` holds release tooling.

## Conventions

- Python >= 3.11, fully typed, line length 110 (ruff: E, F, I, UP, B).
- Global runtime state is created lazily and cleared by the module's `reset_state()`; `tests/conftest.py`
  calls it around every test.
- No global asyncio primitives (locks, events, queues) at module level.
- New tool parameters are keyword-only.
- Tests are offline by default and use the `fake_upstream` fixture; tests that hit the real API are
  marked `live` and are skipped without a key.

## Invariants (do not change without an explicit maintainer decision)

- Tool names and error codes are stable public API.
- `SYSTEM_PROMPT` is frozen.
- The result dedupe heuristic is unchanged.
- The query rewrite model is `deepseek-flash` only.
- SSRF checks are static: never resolve DNS locally (the page is fetched remotely by DeepSeek).
- Errors are never cached.
- No telemetry.
- Keep the legacy opencode `auth.json` credential fallback.

## Commits & releases

- Conventional Commits; subject at most 72 characters, imperative mood; the body explains why.
- One logical change per commit. Never commit with failing tests or lint. Stage explicit paths, never `git add -A`.
- Release automation is release-please in pre-1.0 mode:
  - `fix` and `feat` bump the patch version; `feat!` or a `BREAKING CHANGE` footer bumps the minor.
  - `docs`, `refactor`, `perf` and `deps` appear in the changelog without a bump.
  - `chore`, `ci`, `test`, `build` and `style` are hidden from the changelog.
- PR titles must also be Conventional Commits, because squash merges use them as the commit message.
- Versions are bumped only by release-please. Never edit the version in `pyproject.toml` or `server.json` by hand.
- `CHANGELOG.md` entries after 0.2.0 are generated; do not edit them manually.

## Security

- Never read, print or commit `.env` or API keys.
- Do not log secrets; keep the redaction in `redact.py` applied to anything leaving the process.
- Report vulnerabilities privately; see `SECURITY.md`.
