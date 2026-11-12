# AGENTS.md

## Project overview

`deepseek-search-everywhere-mcp` is a Python MCP server exposing 9 tools over the DeepSeek Search & Fetch API
(stdio by default, optional bearer-protected HTTP): `deepseek_search`, `deepseek_search_many`, `deepseek_fetch`,
`deepseek_fetch_many`, `deepseek_extract`, `deepseek_extract_many`, `deepseek_research` (cited multi-source
reports in four depth tiers, run as background jobs), `deepseek_check_research` (collects a running research
job) and `deepseek_check_balance`.

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
scripts/smoke_docker.sh                     # build, verify non-root UID and test HTTP tools in Docker
```

## Code layout (`src/deepseek_search_everywhere_mcp/`)

- `server.py`: FastMCP server; the 9 tool definitions, result shaping, research job orchestration, chat gateway.
- `deep_research.py`: `DEPTH_CONFIG` tiers, research job store, decomposition and synthesis prompts, citation
  normalization, source ranking and the synthesis output filter.
- `extract.py`: extraction prompts, JSON handling and evidence verification.
- `snippets.py`: lexical snippet and section extraction for `query` and `section` selectors in fetch tools.
- `filters.py`: search filter compilation and `fit_query` (128-character query budget).
- `sanitize.py`: strips hidden Unicode and neutralizes markdown images and unsafe links in untrusted content.
- `egress.py`: egress guard for outgoing queries and URLs (secret patterns, domain lists, entropy check).
- `budget.py`: daily spend budget with reserve/commit/release and a cross-process file lock.
- `transport.py`: HTTP calls to the DeepSeek API; credentials, retries bounded by a deadline, bounded response reads.
- `cache.py`: TTL + LRU response cache with single-flight.
- `ratelimit.py`: per-endpoint token-bucket limiter (search, fetch, chat).
- `settings.py`: `Settings` defaults, LLM pricing and the environment switches.
- `urls.py`: static URL validation (SSRF abuse filter) for fetch tools.
- `errors.py`: error envelopes and the closed set of error codes (`CLOSED_ERROR_CODES`).
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
- Never commit implementation plans, roadmaps, or temporary design documents into the repository;
  keep plans strictly as agent session artifacts or local untracked scratchpads.

## Invariants (do not change without an explicit maintainer decision)

- Tool names, parameters, envelope fields and error codes are stable public API. The 3.0 release
  (maintainer decision, 2026-11-12) deliberately broke this invariant once (renamed and removed tools,
  parameters, envelope fields and error codes). From 3.0 on the invariant holds again: any further rename
  or removal is a breaking change that needs an explicit maintainer decision.
- Error codes come from the closed set in `errors.py` (`CLOSED_ERROR_CODES`); new codes are added there first.
- Server-side LLM calls never use tool calling.
- No secrets in prompts (API keys, tokens and credentials never enter any LLM prompt, query or URL).
- Every chat-completions payload sets `thinking` explicitly (`deepseek-flash` defaults to thinking on).
- Every path that returns untrusted web content goes through the sanitizer (`sanitize.py`) and carries
  `untrusted_content: true`; model-written output derived from it also carries `derived_from_untrusted: true`.
- README shows DeepSeek off-peak prices only, with the off-peak footnote under every price table or figure.
- `SYSTEM_PROMPT` is frozen.
- The result dedupe heuristic is unchanged.
- The query rewrite model is `deepseek-flash` only.
- SSRF checks are static: never resolve DNS locally (the page is fetched remotely by DeepSeek).
- Errors are never cached.
- No telemetry.
- Keep the legacy opencode `auth.json` credential fallback.

## Branching & workflow

- Never push or force-push directly to `main`; `main` is protected and immutable. All changes must arrive via Pull Requests.
- Feature branches use conventional prefixes: `feat/<topic>`, `fix/<topic>`, `docs/<topic>`, `chore/<topic>`.
- Never use raw `git push --force`. When rebasing a personal feature branch, use `git push --force-with-lease`.
- Run local checks before opening or requesting review on a PR:
  ```bash
  ruff check . && ruff format --check . && pytest -q
  ```
- Automated `release-please` PRs (e.g. `chore(main): release X.Y.Z`) must not be manually edited; merge them directly to trigger publication.

## Commits & releases

- Conventional Commits; subject at most 72 characters, imperative mood; the body explains why.
- One logical change per commit. Never commit with failing tests or lint. Stage explicit paths, never `git add -A`.
- Release automation is release-please with standard semantic versioning (1.0 and later):
  - `fix` bumps the patch version and `feat` bumps the minor.
  - A breaking change must be marked with `!` after the type (`feat!:`) or a `BREAKING CHANGE:` footer,
    and it implies a major release. Tool names, parameters, envelope fields and error codes are public
    API: renaming or removing any of them, or changing their meaning, is a breaking change.
  - `docs`, `refactor`, `perf` and `deps` appear in the changelog without a bump.
  - `chore`, `ci`, `test`, `build` and `style` are hidden from the changelog.
- PR titles must also be Conventional Commits, because squash merges use them as the commit message.
- Versions are bumped only by release-please. Never edit the version in `pyproject.toml` or `server.json` by hand.
- `CHANGELOG.md` entries after 0.2.0 are generated; do not edit them manually.

## Security

- Never read, print or commit `.env` or API keys.
- Do not log secrets; keep the redaction in `redact.py` applied to anything leaving the process.
- Report vulnerabilities privately; see `SECURITY.md`.
