<p align="center">
  <img src="assets/banner.jpg" alt="DeepSeek Search Everywhere MCP banner" width="100%">
</p>

# DeepSeek Search Everywhere MCP

<p align="center">
  <a href="https://pypi.org/project/deepseek-search-everywhere-mcp/"><img src="https://img.shields.io/pypi/v/deepseek-search-everywhere-mcp.svg" alt="PyPI version"></a>
  <a href="https://pypistats.org/packages/deepseek-search-everywhere-mcp"><img src="https://img.shields.io/pypi/dw/deepseek-search-everywhere-mcp?color=blue&label=pypi%20downloads" alt="PyPI downloads per week"></a>
  <a href="https://pypi.org/project/deepseek-search-everywhere-mcp/"><img src="https://img.shields.io/pypi/pyversions/deepseek-search-everywhere-mcp" alt="Supported Python versions"></a>
  <a href="https://registry.modelcontextprotocol.io"><img src="https://img.shields.io/badge/MCP%20Registry-listed-brightgreen.svg" alt="Listed in the official MCP Registry"></a>
  <a href="https://github.com/radoxtech/deepseek-search-everywhere-mcp/actions/workflows/ci.yml"><img src="https://github.com/radoxtech/deepseek-search-everywhere-mcp/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://glama.ai/mcp/servers/radoxtech/deepseek-search-everywhere-mcp"><img src="https://glama.ai/mcp/servers/radoxtech/deepseek-search-everywhere-mcp/badges/score.svg" alt="Glama score"></a>
</p>

**DeepSeek Search Everywhere** is a [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that gives AI agents web search, page fetch (web scraping to clean Markdown), structured data extraction and deep research with numbered citations, all through the DeepSeek Search & Fetch API. It runs over stdio or authenticated HTTP and works with Claude Code, Claude Desktop, Cursor, Cline, Windsurf, Zed, OpenCode and any other MCP client.

You need one credential, a DeepSeek API key, and you pay DeepSeek directly per request: there is no separate subscription. It is an alternative to Tavily, Perplexity, Firecrawl and Exa for agents that need live web access, and about 10x cheaper than Tavily for basic search; see [Comparison](#comparison-with-tavily) for the numbers.

<!-- mcp-name: io.github.radoxtech/deepseek-search-everywhere-mcp -->

> Not affiliated with DeepSeek. This is an independent, community-maintained project.

## Demo

<p align="center">
  <img src="assets/demo.gif" alt="OpenCode terminal session calling deepseek_search, deepseek_fetch_many, deepseek_extract and deepseek_research" width="100%">
</p>
<p align="center">
  <em>OpenCode calling <code>deepseek_search</code>, <code>deepseek_fetch_many</code>, <code>deepseek_extract</code> and <code>deepseek_research</code>. The same tools work in every MCP client.</em>
</p>

## Key features

- **Web search** with snippets or full page text, domain and date filters, language selection and optional query rewrite.
- **Page fetch to Markdown** for one URL or up to 10 in parallel, with section, snippet, offset and find-text selectors for long pages.
- **Structured extraction**: pull named fields or a JSON schema from a page; each value must be backed by a verbatim quote from the page or it is returned as `null`.
- **Deep research with citations**: four depth tiers and `academic`/`technical` source modes produce a report with `[n]` citations mapped to a `sources` array, with number verification. The server writes the report, or your agent writes it from the ranked sources (`synthesis="client"`) and the server verifies it. See [Deep research](#deep-research).
- **Cost reporting**: every response carries `cost_usd` and, where an LLM is involved, `cost_breakdown`; an optional daily budget blocks overspending.
- **Hardened against prompt injection**, with an egress guard, static SSRF filtering and no telemetry. See [Security](#security).
- **Small footprint**: pure Python, no local browser, `uvx` zero-install, 9 tools.

## Tools

| Tool | What it does | Key parameters |
|---|---|---|
| `deepseek_search` | Search the web for one query; returns ranked results with snippets, or full page text | `query`, `count` (1-50, default 5), `content` (`snippets` or `full`), `lang`, `include_domains`, `exclude_domains`, `after`, `before`, `exclude_terms`, `exact_phrases`, `rewrite`, `max_total_chars` (default 20000), `refresh` |
| `deepseek_search_many` | Run 1-5 searches concurrently; partial results are returned if the deadline hits, and unique queries past the fifth are skipped and listed in `skipped_queries` | `queries`, plus the filters of `deepseek_search`, `max_total_chars` (default 20000), `deadline` |
| `deepseek_fetch` | Fetch one known URL as Markdown | `url`, `max_chars` (default 20000), one optional selector: `section`, `query` (+`max_snippets`), `offset`+`limit`, or `find_text`; `strip_boilerplate`, `fallback_to_search`, `refresh` |
| `deepseek_fetch_many` | Fetch 1-10 URLs concurrently (up to 5 at a time) | `urls`, `max_chars_per_url` (default 4000), `max_total_chars` (default 20000), `query`, `deadline` |
| `deepseek_extract` | Extract structured JSON from one page, with evidence quotes | `url`, `fields` or `schema`, `instruction`, `max_chars` (default 40000) |
| `deepseek_extract_many` | Extract structured JSON from 1-10 pages concurrently | `urls`, `fields` or `schema`, `instruction`, `max_chars_per_url`, `max_total_chars`, `deadline` |
| `deepseek_research` | Multi-angle research: decomposes the topic, searches, reads sources and writes a cited report | `topic` (max 500 chars), `depth` (`low`, `medium`, `high`, `max`; default `medium`), `sources` (`any`, `academic`, `technical`; default `any`), `lang`, `unverified` (`flag` or `drop`; default `flag`), `synthesis` (`server` or `client`; default `server`), `strip_boilerplate`, `refresh` |
| `deepseek_check_research` | Continue a research job: poll it, read the sources of a `synthesis="client"` job page by page, or submit your report for verification | `job_id`, `page`, `report` |
| `deepseek_check_balance` | Read-only: DeepSeek account balance and session metrics | none |

Details that matter when calling the tools:

- `query` and each entry of `queries` are limited to 128 characters and accept `site:`, `-site:`, `"phrase"`, `lang:`, `after:` and `before:` operators. The explicit filter parameters (`include_domains`, `exclude_domains`, `exclude_terms`, `exact_phrases`, `after`, `before`) are compiled into operators and budget-checked against that limit together with the query. Dates use `YYYY-MM-DD`.
- `content="snippets"` (default) returns a short snippet per result; `content="full"` returns page text and fetches pages whose search content is missing, one extra request per fetched page, which is reflected in `requests_used` and `cost_usd`.
- `deepseek_fetch` selectors are mutually exclusive: combining `section`, `query`, `offset`/`limit` and `find_text` returns `bad_request`. `query` returns focused excerpts around matching terms (lexical matching, no LLM call). `offset`/`limit` page through long text and report `total_chars`.
- `deepseek_fetch_many` skips blanks, fetches duplicate URLs once, keeps input order and reports a failing URL as an item with `ok: false`; the top-level envelope sets `partial: true`.
- `deepseek_extract` takes either `fields` (list of names) or `schema` (JSON schema), never both. Fields whose evidence quote cannot be found in the page are set to `null` with a note; `extraction_truncated` is raised when the model output was cut off.
- `lang` is an ISO code such as `en`, `de`, `pt-BR`. `rewrite=true` translates the query to English with `deepseek-flash` before searching and is skipped when `lang` is set.
- `strip_boilerplate=true` removes known site interface text (for example the localized Reddit UI and English cookie-consent dialogs at the start of a page) without touching article text or code blocks. Search results always get this treatment.
- `fallback_to_search=true` (fetch tools) retrieves a document through the search index when a direct fetch returns 404 for a PDF or Reddit URL; only a result for the same page is accepted, and the result is marked `"source": "search_index"`.

Example calls:

```text
deepseek_search(query="rust async runtime comparison", count=5)
deepseek_search(query="gobierno de España presupuestos 2026", lang="es")
deepseek_search(query="tokio runtime", include_domains=["docs.rs"], content="full")
deepseek_fetch(url="https://docs.rs/tokio/latest/tokio/", max_chars=15000)
deepseek_fetch(url="https://example.com/long-article", query="rate limits")
deepseek_fetch_many(urls=["https://example.com/a", "https://example.com/b"])
deepseek_extract(url="https://news.ycombinator.com", fields=["top_story", "points", "author"])
deepseek_research(topic="Raft vs Paxos consensus: trade-offs in practice", depth="high")
deepseek_check_research(job_id="<job_id from a running response>")
deepseek_check_balance()
```

## How agents should use this server

Pick the tool by what the task needs:

| Task | Tool |
|---|---|
| A question that needs several sources: comparison, "how does X work", state of the art, pros and cons, a decision | `deepseek_research` (`depth` `low` to `max`; `sources="academic"` or `"technical"` when it fits) |
| One fact, a recent event, a page to find | `deepseek_search` (snippets first, `content="full"` only when the page text is needed) |
| Several independent searches at once | `deepseek_search_many` (up to 5 queries) |
| A known URL | `deepseek_fetch`; pass `query=` to get only the relevant excerpts |
| Several known URLs | `deepseek_fetch_many` |
| Specific fields or a JSON schema from a page | `deepseek_extract` or `deepseek_extract_many` |
| Account balance and session usage | `deepseek_check_balance` |

Research jobs have three states, and the response says which one applies:

- `status: "done"`: the report is in `report` and its `[n]` citations are the `id`s in `sources`.
- `status: "running"`: call `deepseek_check_research(job_id)` until it is done; do not start the same research again.
- `status: "awaiting_report"` (only with `synthesis="client"`): read the sources (`context`, further pages with `page=`), write the report and send it with `deepseek_check_research(job_id, report=...)`.

Common mistakes to avoid: a query plus its filters must fit in 128 characters; the `deepseek_fetch` selectors (`section`, `query`, `offset`/`limit`, `find_text`) cannot be combined; `deepseek_search_many` runs at most 5 unique queries and lists the rest in `skipped_queries`; check `partial` on batch results, because `ok` stays `true` when single items fail. The [error codes](#error-codes) table says which errors are worth retrying.

## Quick start

1. Create a DeepSeek API key at [platform.deepseek.com](https://platform.deepseek.com) and add a small balance.
2. Install [uv](https://docs.astral.sh/uv/) (or use `pip`; Python 3.11 or newer is required).
3. Add the server to your MCP client. The minimal configuration is:

```json
{
  "mcpServers": {
    "deepseek-search-everywhere": {
      "command": "uvx",
      "args": ["deepseek-search-everywhere-mcp"],
      "env": { "DEEPSEEK_API_KEY": "sk-..." }
    }
  }
}
```

4. Optional: verify the key and connectivity from a terminal. This prints your balance and exits:

```bash
export DEEPSEEK_API_KEY=sk-...
uvx deepseek-search-everywhere-mcp --check
```

The package is on [PyPI](https://pypi.org/project/deepseek-search-everywhere-mcp/) and in the
[official MCP Registry](https://registry.modelcontextprotocol.io/v0/servers?search=deepseek-search-everywhere) as
`io.github.radoxtech/deepseek-search-everywhere-mcp`. The console script is `deepseek-search-everywhere-mcp`;
`python -m deepseek_search_everywhere_mcp` is equivalent.

```bash
pip install deepseek-search-everywhere-mcp              # or: uvx deepseek-search-everywhere-mcp
pip install "deepseek-search-everywhere-mcp[http]"      # adds uvicorn for the HTTP transport
uvx --from git+https://github.com/radoxtech/deepseek-search-everywhere-mcp deepseek-search-everywhere-mcp  # latest main
```

## Install in your MCP client

All stdio examples use `uvx`, so nothing is installed globally. Replace `sk-...` with your key.

### Claude Code

```bash
claude mcp add deepseek-search-everywhere -e DEEPSEEK_API_KEY=sk-... -- \
  uvx deepseek-search-everywhere-mcp
```

### Claude Desktop, Cursor, Cline, Roo Code, Windsurf and other `mcpServers` clients

These clients use the same `mcpServers` block as the [Quick start](#quick-start); only the file differs:

| Client | Configuration file |
|---|---|
| Claude Desktop | `claude_desktop_config.json` (Settings, Developer, Edit Config) |
| Cursor | `.cursor/mcp.json` in the project, or `~/.cursor/mcp.json` for all projects |
| Cline, Roo Code (VS Code) | `cline_mcp_settings.json` (MCP settings) |
| Windsurf | `~/.codeium/windsurf/mcp_config.json` |
| Antigravity, Gemini CLI and other stdio clients | the client's MCP server configuration |

With the package installed locally (`pip install deepseek-search-everywhere-mcp`), `"command": "deepseek-search-everywhere-mcp"` without `args` works as well.

### Zed

In Zed `settings.json`, under `context_servers`:

```json
{
  "context_servers": {
    "deepseek-search-everywhere": {
      "source": "custom",
      "command": "uvx",
      "args": ["deepseek-search-everywhere-mcp"],
      "env": { "DEEPSEEK_API_KEY": "sk-..." }
    }
  }
}
```

### OpenCode

In `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "deepseek-search-everywhere": {
      "type": "local",
      "command": ["uvx", "deepseek-search-everywhere-mcp"],
      "environment": { "DEEPSEEK_API_KEY": "{env:DEEPSEEK_API_KEY}" },
      "enabled": true
    }
  }
}
```

### Streamable HTTP

For clients and platforms that connect over HTTP. A bearer token is always required and the server refuses to start without `MCP_BEARER_TOKEN`.

```bash
pip install "deepseek-search-everywhere-mcp[http]"
export DEEPSEEK_API_KEY=sk-...
export MCP_BEARER_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
deepseek-search-everywhere-mcp --transport streamable-http   # http://127.0.0.1:8000/mcp
```

Options: `--host` (default `127.0.0.1`), `--port` (default `8000`), `--path` (default `/mcp`). The token is read from the environment only, never from the command line. The server is stateless and answers with JSON (no SSE streams). Client entry:

```json
{
  "mcpServers": {
    "deepseek-search-everywhere": {
      "url": "http://127.0.0.1:8000/mcp",
      "headers": { "Authorization": "Bearer <MCP_BEARER_TOKEN>" }
    }
  }
}
```

Claude Code over HTTP:

```bash
claude mcp add --transport http deepseek-search-everywhere http://127.0.0.1:8000/mcp \
  --header "Authorization: Bearer $MCP_BEARER_TOKEN"
```

The server speaks plain HTTP; when binding beyond loopback, put TLS in front of it (see [SECURITY.md](SECURITY.md)). HTTP mode is single-process: research jobs, caches and the rate limiter live in that process.

### Docker

A container image is published on GitHub Container Registry. It runs as an unprivileged user (UID 10001).

```bash
docker pull ghcr.io/radoxtech/deepseek-search-everywhere-mcp:latest
```

Stdio mode for desktop clients:

```json
{
  "mcpServers": {
    "deepseek-search-everywhere": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "DEEPSEEK_API_KEY=sk-...",
        "ghcr.io/radoxtech/deepseek-search-everywhere-mcp:latest"
      ]
    }
  }
}
```

Remote HTTP hosting:

```bash
docker run -d --name deepseek-mcp -p 8000:8000 \
  -e DEEPSEEK_API_KEY="sk-..." \
  -e MCP_BEARER_TOKEN="your-secret-token" \
  ghcr.io/radoxtech/deepseek-search-everywhere-mcp:latest \
  --transport streamable-http --host 0.0.0.0 --port 8000
```

## Deep research

`deepseek_research(topic, depth=...)` turns a question into a cited report. It decomposes the topic into subqueries, searches them concurrently, ranks and de-duplicates sources (at most 2 per registered domain, with hosts from the source denylist skipped), reads the top pages, and has `deepseek-flash` write a report whose `[n]` citations match the ids in the returned `sources` array. By default synthesis runs on the server, so the consuming agent receives a short report and a list of sources instead of tens of thousands of characters of page text; with `synthesis="client"` the agent writes the report itself (see [Client-side synthesis](#client-side-synthesis)).

### Depth tiers

| | `low` | `medium` (default) | `high` | `max` |
|---|---|---|---|---|
| Subqueries | 2 | 3 | 4 | 4, plus a gap-analysis round |
| Results per subquery | 6 | 8 | 10 | 10 |
| Sources read (K) | 6 | 10 | 16 | 24 |
| Characters per source page (cap) | 15,000 | 20,000 | 25,000 | 30,000 |
| Total source context | 60,000 chars | 150,000 chars | 300,000 chars | 500,000 chars |
| Reasoning in synthesis | off | on, effort `low` | on, effort `high` | on, effort `max` |
| Synthesis `max_tokens` | 16,384 | 32,768 | 65,536 | 131,072 |
| Report length target | about 400 words | about 800 | about 1,200 | about 1,800 |
| Report length ceiling (+20%) | 480 | 960 | 1,440 | 2,160 |
| Total deadline | 60 s | 90 s | 150 s | 240 s |

When to pick each tier:

- **`low`**: a quick factual overview or a sanity check with a handful of sources; fastest and cheapest.
- **`medium`**: the default for most questions: comparisons, "how does X work", state of a technology.
- **`high`**: decisions that depend on several viewpoints, such as architecture choices, vendor evaluations or regulatory questions.
- **`max`**: broad or contested topics where coverage matters more than speed; the extra round looks for gaps in the first round and searches again.

If the model's reasoning uses up its token allowance or returns an empty answer, the server retries once with reasoning off; if that also fails, the envelope has `synthesized: false` and returns source snippets only. An over-length report gets one cheap condensing pass.

### Long-running research

Many MCP clients cancel a tool call after about 60 seconds, which is shorter than the `high` and `max` deadlines. `deepseek_research` therefore never blocks past an inline wait of 45 seconds:

1. The research runs as a background job inside the server process, not inside the tool call. A client timeout or a cancelled call does not stop the job.
2. If the report is ready within 45 s, `deepseek_research` returns it inline with `status: "done"` and `delivery: "inline"`.
3. Otherwise it returns `ok: true`, `status: "running"`, a `job_id`, `elapsed_ms`, a `progress` object (`phase`, `step`, `of`, an optional `detail` for a synthesis `retry` or `condense` sub-step, `phase_elapsed_ms` spent in the current phase and `deadline_ms`, the depth's hard time cap) and a `next_action` telling the agent to call `deepseek_check_research` with that `job_id` and not to start a new research.
4. `deepseek_check_research(job_id)` waits up to another 45 s and returns either the same final envelope (`status: "done"`, `delivery: "polled"`) or `status: "running"` again. Repeat until it is done. An unknown or expired id returns `job_not_found`; start the research again in that case (the job may have been lost on a server restart).

Behavior to know:

- An identical request (same topic, `depth`, `lang`, `sources`, `unverified` and `synthesis`) made while a job is running joins that job and gets the same `job_id`, so a retry never doubles the cost.
- A finished result is reused for 30 minutes at zero cost (`cached: true`, `cost_usd: 0`); pass `refresh=true` to run again.
- At most 3 research jobs run at the same time; further requests return `rate_limited_local` with advice to collect a running job with `deepseek_check_research` or retry after one finishes (usually 30-180 s).
- Jobs live in memory in one server process. They are cancelled when the server shuts down.

### Client-side synthesis

Pass `synthesis="client"` when your agent should write the report with its own model, for example to follow a specific structure, analyse the evidence more deeply or apply its own judgement. The server still does the research work: decomposition, search, ranking and domain diversity, gap fetching and the choice of sources.

1. `deepseek_research(topic, depth, ..., synthesis="client")` runs the job as usual but stops before the server's synthesis. It returns `status: "awaiting_report"` with `synthesis_instructions` (citation rules and the word budget of the depth), the first page of `context` (sources with `id`, `title`, `url`, `date`, `doi`, `pmid`, `pmcid`, `access`, `source_type`, `authors` and their text), `context_page`, `context_pages` and a `next_action`.
2. Read further pages with `deepseek_check_research(job_id, page=N)`. A page holds about 40,000 characters of source text, so each response stays under the tool-output limits of MCP clients; a longer source is split into parts (`part`, `parts`). The cost of gathering the sources is reported once, with the first delivery of page 1; later pages, repeats and report verification cost nothing. Pages stay readable after you send a report, so you can check a source and send a corrected report.
3. Send the report with `[id]` citations: `deepseek_check_research(job_id, report="...")`. The server applies the same checks as to its own reports: links outside the source list, images, raw HTML and instructions aimed at an agent are removed, citations to unknown ids are dropped, and numbers missing from the cited source are marked `[unverified]` (or removed with `unverified="drop"`). Your report is never shortened; a report over the word ceiling only gets a note. The verified envelope has `status: "done"` and `delivery: "client"`, and becomes the job's result. You can send a corrected report again; the latest one wins.

The model cost moves to your side: the source text of a job is about 60,000 characters at `low` and up to 500,000 at `max`, so `low` or `medium` suit client synthesis best. The job keeps its sources in memory while you write (each call refreshes its 30-minute lifetime); a server restart loses them, and a new `deepseek_research` call starts again.

### Source preferences and metadata

`deepseek_research` supports a `sources` parameter (`"any"`, `"academic"`, or `"technical"`) to prioritize domain authorities and tailor subquery phrasing:

- **`"any"`** (default): General web research with authority-aware ranking across all domains.
- **`"academic"`**: Prioritizes peer-reviewed scholarly literature and research repositories (PubMed, PMC, Europe PMC, arXiv, bioRxiv/medRxiv, Nature, ScienceDirect, Springer, Wiley, Oxford Academic, IEEE, PLOS, Frontiers, and `.edu`/`.ac.*` repositories), granting bonuses to sources reporting DOI or PMID identifiers. Subqueries are phrased to target trials, meta-analyses, and systematic reviews.
- **`"technical"`**: Prioritizes official engineering documentation (`docs.*`, `learn.microsoft.com`, `kubernetes.io`, `docs.aws.amazon.com`, `cloud.google.com/docs`, `readthedocs.io`, MDN, etc.), specifications and RFCs (IETF, W3C, RFC Editor), GitHub/GitLab repositories, release notes, changelogs, and technical Q&A (Stack Overflow, Stack Exchange). Subqueries target official docs, changelogs, and issues.

Every entry in the returned `sources` array provides rich metadata:

- **`id`**: Integer citation ID matching `[n]` citation markers in `report`.
- **`title`**: Sanitized page title, free of HTML markup and formatting artifacts.
- **`url`**: Clean canonical URL. Query trackers, signed URL parameters, and expiring access tokens are automatically stripped. Library proxy hostnames (such as EZproxy and libproxy) are canonicalized back to their upstream publisher hosts (e.g. `journals-lww-com.libproxy.ucl.ac.uk` becomes `journals.lww.com`).
- **`date`**: Publication date (`YYYY-MM-DD`): from the URL path when it contains a full date, otherwise from the search index, otherwise from the start of the page text; `null` when none is found. Search results also report where it came from in `date_source`.
- **`doi`**: Digital Object Identifier (e.g. `10.1000/xyz.123`) extracted from URL or scholarly header metadata (`null` if absent).
- **`pmid`**: PubMed ID extracted from URL or scholarly header metadata (`null` if absent).
- **`pmcid`**: PubMed Central ID (e.g. `PMC1234567`) extracted from URL or scholarly header metadata (`null` if absent).
- **`authors`**: Author names read from an explicit `Authors:` line or a `By ...` byline near the top of the source text, or `null` when there is none. Affiliations are not extracted.
- **`access`**: Source text coverage: `"full_text"` (page fetched, or at least 2,000 characters of page text from the search index), `"abstract"` (shorter search text of at least 200 characters) or `"snippet_only"`.
- **`source_type`**: Heuristic content category (`meta-analysis`, `systematic_review`, `rct`, `study`, `guideline`, `preprint`, `review`, `documentation`, `specification`, `repository`, `qa`, `news`, `blog`, `vendor`, `video`, `other`).

### Number grounding verification

To protect against numerical hallucination in synthesized research reports, every sentence containing `[n]` citations is parsed for numbers (including percentages, decimal values, ranges, and negative quantities) and checked against the source texts cited by that sentence:

- **Verifiable facts**: Numbers found verbatim in the cited sources remain unmarked.
- **Unverified numbers**: Sentences containing numbers that do not appear in the cited source text are marked with a trailing ` [unverified]` tag. With `unverified="drop"` such sentences are removed from the report instead; they stay listed in `unverified_numbers`. The mode is part of the job key, so a flagged and a dropped run of the same topic are separate jobs.
- **Envelope fields**: The top-level research envelope includes `unverified_numbers` (a list of `{ "number", "citation", "sentence" }` entries) and `unverified_number_count`, alongside human-readable notices in `notes`.

## Costs

Search and Fetch are billed by DeepSeek per successful request, not per returned result, so `count` does not change the price of a call. Extraction and research add `deepseek-flash` token costs on top of the requests they make. Cache hits cost nothing.

| Item | Price |
|---|---|
| Search and Fetch API | $0.75 per 1,000 successful requests ($0.00075 each) |
| `deepseek-flash` input, cache hit | $0.003 per 1M tokens |
| `deepseek-flash` input, cache miss | $0.15 per 1M tokens |
| `deepseek-flash` output | $0.60 per 1M tokens |

*Prices shown are DeepSeek off-peak rates as of 2026-11-12; see https://api-docs.deepseek.com/quick_start/pricing.*

Measured cost and wall time of one `deepseek_research` call per tier (live runs):

| Depth | Typical cost | Typical time |
|---|---|---|
| `low` | about $0.003 | about 6 s |
| `medium` | about $0.008 | about 18-29 s |
| `high` | about $0.012 | about 39 s |
| `max` | about $0.024-0.031 | about 50-110 s |

*Prices shown are DeepSeek off-peak rates as of 2026-11-12; see https://api-docs.deepseek.com/quick_start/pricing.*

Other tools: `deepseek_search` and `deepseek_fetch` cost one request each ($0.00075); `content="full"` and `deepseek_fetch_many` add one request per page fetched; `deepseek_extract` costs about $0.002 per page (one fetch plus the LLM call). `rewrite=true` on the search tools adds one small `deepseek-flash` call.

*Prices shown are DeepSeek off-peak rates as of 2026-11-12; see https://api-docs.deepseek.com/quick_start/pricing.*

Every response reports its own `cost_usd`, and tools that call an LLM add `cost_breakdown` with `search`, `fetch` and `llm` parts. Set `DEEPSEEK_DAILY_BUDGET_USD` to cap daily spend (see [Configuration](#configuration)).

## Comparison with Tavily

Figures as of 2026-11-12. Tavily pay-as-you-go is $0.008 per credit; a basic search costs 1 credit, an advanced search 2 credits, and a basic extract 1 credit per 5 URLs ([Tavily API credits](https://docs.tavily.com/documentation/api-credits)). Tavily also has a free tier of 1,000 credits per month, which covers light use at no cost.

| Operation | Tavily (pay-as-you-go) | This server (Search and Fetch API) | Ratio |
|---|---|---|---|
| Basic search | $0.008 | $0.00075 | about 10.7x cheaper |
| Advanced search | $0.016 | $0.00075 | about 21x cheaper |
| Page fetch, per URL (basic extract) | $0.0016 | $0.00075 | about 2.1x cheaper |

*Prices shown are DeepSeek off-peak rates as of 2026-11-12; see https://api-docs.deepseek.com/quick_start/pricing.*

The comparison covers raw retrieval prices only, not result quality; extraction and research add `deepseek-flash` token costs ([Costs](#costs)). Tavily, Perplexity, Firecrawl and Exa have features this server does not, such as site crawling or hosted answer engines.

## Security

Web pages, search snippets and titles are attacker-controlled. The server treats everything it retrieves as untrusted and applies these controls (full threat model in [SECURITY.md](SECURITY.md)):

- **Untrusted-content handling.** Hidden Unicode (tag characters, bidi controls, zero-width characters) is stripped from content, snippets, titles, reports and extracted values. Markdown images are rewritten to plain `[image: alt](url)` text and links with non-http(s) targets are dropped. Every response that carries web data sets `untrusted_content: true`; research and extraction output, which is model-written from untrusted pages, also sets `derived_from_untrusted: true`.
- **Injection-resistant synthesis.** The server's own LLM calls have no tool calling and no secrets in their prompts. Sources are wrapped in per-call random delimiters and declared to be data, and the report is filtered afterwards: links that are not in the source list, images, raw HTML and sentences addressed to an AI agent are removed (counted in `injection_suspected`).
- **Egress guard.** Outgoing search queries and fetch URLs that contain secret-like data (`sk-` keys, bearer tokens, JWTs, AWS access keys, `api_key=`/`token=`/`password=` pairs, URL credentials) are rejected with `egress_blocked`. Optional domain allow and deny lists and a high-entropy check are available through [environment variables](#configuration).
- **Static SSRF filter.** Non-http(s) URLs return `bad_url`; internal hostnames, private and loopback addresses and URLs with embedded credentials return `ssrf_blocked`. The server never resolves DNS locally; DeepSeek performs the fetch, so this is a filter and not network isolation.
- **No telemetry.** The server does not phone home or collect usage data.
- **Data flow.** Search queries, URLs you ask it to fetch, and page content selected for extraction or research are sent to the DeepSeek API; nothing else leaves the machine, and the API key is only sent in the `Authorization` header to the configured base URL. Do not put secrets in queries or URLs.
- **HTTP transport.** Every request needs the bearer token (constant-time comparison, never logged). Bind to loopback or put TLS in front of the server.

The consuming agent must still treat returned content as untrusted: no filter can guarantee that a page contains no instructions. Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

## Configuration

The server reads configuration from environment variables. It does not read `.env` files itself; export the variables or set them in your MCP client's `env` block.

**Required**

| Variable | Purpose |
|---|---|
| `DEEPSEEK_API_KEY` | Your DeepSeek platform key |
| `MCP_BEARER_TOKEN` | HTTP transport only: token clients send as `Authorization: Bearer ...`; the server refuses to start HTTP without it |

**Optional**

| Variable | Default | Purpose |
|---|---|---|
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API base URL |
| `DEEPSEEK_MCP_CACHE` | `1` | `0` disables the in-memory response caches |
| `DEEPSEEK_MCP_LOG_LEVEL` | `WARNING` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `DEEPSEEK_DAILY_BUDGET_USD` | unlimited | Daily spend cap in USD, e.g. `5.00` |
| `DEEPSEEK_STATE_DIR` | see below | Directory of the budget state file |
| `DEEPSEEK_SOURCE_DENYLIST` | unset | Hosts research never uses as sources (library-proxy hosts are rewritten to the publisher); `none` disables |
| `DEEPSEEK_FETCH_ALLOW_DOMAINS` | unset | Fetch only these domains |
| `DEEPSEEK_FETCH_DENY_DOMAINS` | unset | Never fetch these domains |
| `DEEPSEEK_EGRESS_ENTROPY_CHECK` | off | `1` also blocks long high-entropy tokens |
| `DEEPSEEK_AUTH_JSON` | see below | Legacy key file, used only without `DEEPSEEK_API_KEY` |

Details:

- **Daily budget** covers Search, Fetch and LLM costs and resets per UTC day; once reached, calls fail with `budget_exceeded`. The state lives in `budget.json` under `DEEPSEEK_STATE_DIR` (default `~/.cache/deepseek-search-everywhere-mcp`).
- **Domain lists** (`DEEPSEEK_SOURCE_DENYLIST`, `DEEPSEEK_FETCH_ALLOW_DOMAINS`, `DEEPSEEK_FETCH_DENY_DOMAINS`) are comma-separated; an entry matches the host and its subdomains, and an entry containing `*` is a glob matched against the whole host. `DEEPSEEK_SOURCE_DENYLIST` defaults to empty because library-proxy hosts (EZproxy, libproxy) are automatically rewritten to canonical publisher URLs instead of rejected.
- **Entropy check** (`1`, `true` or `yes`) rejects queries and URL paths that contain a token longer than 32 characters with high entropy.
- **Legacy key file**: `DEEPSEEK_AUTH_JSON` points to an opencode credentials file (`{"deepseek": {"key": "..."}}`); without it the server tries `~/.local/share/opencode/auth.json`.

Command-line options of `deepseek-search-everywhere-mcp`: `--transport {stdio,streamable-http}` (default `stdio`), `--host` (default `127.0.0.1`), `--port` (default `8000`), `--path` (default `/mcp`), `--check` (verify credentials and print the balance; stdio only) and `--version`.

Local limits that are not configurable through the environment:

| | Search | Fetch | Query rewrite |
|---|---|---|---|
| Cache TTL | 5 min | 30 min | 24 h |
| Entry limit | 128 | 128 | 1024 |
| Rate limit (sustained / burst) | 10 / s, 20 | 5 / s, 10 | LLM calls: 10 / s, 20 |

Only successful, non-empty responses are cached and errors are never cached. Identical concurrent calls share one upstream request. `refresh=true` bypasses the cache for one call. If a rate-limit token would take longer than 5 s, the call fails with `rate_limited_local`. HTTP 429 and gateway 5xx responses are retried once within the call's deadline.

## Response format

Every tool returns a JSON object with `ok`.

| Field | Where | Meaning |
|---|---|---|
| `ok` | all | `true` on success, `false` on failure |
| `error` | failures | `{ "code", "message", "status"?, "upstream_code"?, "upstream_message"? }`; `code` is always one of the [closed error codes](#error-codes) |
| `elapsed_ms` | all | wall time of the call in milliseconds |
| `cost_usd` | retrieval tools | cost of the call in USD; `0` for cache hits |
| `cost_breakdown` | extract, research | `{ "search", "fetch", "llm" }` in USD |
| `requests_used` | search, search_many, fetch_many, research | upstream requests actually made |
| `cached`, `cached_count` | retrieval tools | whether the response (or how many items) came from the cache |
| `notes` | success envelopes | human-readable remarks: dedupe counts, trimmed queries, truncation, applied filters |
| `partial`, `failed_count` | batch tools, research | `true` when some items or sources failed; each failed one carries its own `error`. The top-level `ok` stays `true`, so check `partial` |
| `untrusted_content` | tools returning web content | always `true`; the content is data, not instructions |
| `derived_from_untrusted`, `injection_suspected` | research | model-written output built from untrusted pages; number of sentences removed by the output filter |
| `unverified_numbers`, `unverified_number_count` | research | list of numbers not found in cited sources and total unverified count |
| `status`, `delivery`, `job_id`, `progress`, `next_action` | research, check_research | job state: `done` (with `delivery` `inline`, `polled` or `client`), `running`, or `awaiting_report` for client synthesis |
| `synthesis_instructions`, `context`, `context_page`, `context_pages` | research, check_research | client synthesis only: the rules for the report and one page of sources with their text |
| `skipped_queries` | search_many | unique queries past the fifth that were not run |
| `_meta` | retrieval tools | `{ "cost_usd", "untrusted_content" }` for clients that read MCP metadata |

Tool-specific results:

- **Search**: `results[]` with `title`, `url`, `date` (`YYYY-MM-DD`), `date_source` (`url` for a full date in the URL path, which wins; `upstream`; `content`; or `null`), `site`, `snippet` (and `content` with `content="full"`), `authority` and `rerank_score` (upstream fields, not credibility scores).
- **Fetch**: `url`, `title`, `content`, `chars`, `total_chars`, `truncated`, `source` (`fetch` or `search_index`); selectors add `has_more` or section metadata.
- **Extract**: `data` (requested fields or schema), `evidence` (verbatim quotes), `extraction_truncated`.
- **Research**: `topic`, `depth`, `subqueries`, `report` (with `[n]` citations and `[unverified]` sentence markers where numbers lack source grounding), `synthesized`, `sources[]` of `{ id, title, url, date, doi, pmid, pmcid, access, source_type, authors }` where `id` equals `n`, `unverified_numbers`, and `unverified_number_count`. There is no references list inside the report and no page content in the envelope.
- **Balance**: `is_available`, `balances`, and a `session` block with uptime, request and cache counts.

### Error codes

`error.code` comes from a closed set. Upstream DeepSeek errors are mapped to it and the original code and message (redacted) are kept in `error.upstream_code` and `error.upstream_message`.

| Agent action | Codes |
|---|---|
| Retry after a short wait | `rate_limited_local`, `upstream_rate_limited`, `timeout`, `upstream_timeout`, `deadline_exceeded`, `network_error`, `upstream_error`, `request_failed`, `invalid_json`, `internal_error` |
| Fix the call and try again | `bad_request`, `empty_query`, `query_too_long`, `topic_too_long`, `bad_count`, `bad_lang`, `bad_url`, `empty_url`, `url_too_long`, `bad_domain`, `bad_date`, `bad_term`, `bad_phrase`, `filters_too_long` |
| Change the approach: another URL or source, a selector or smaller page (`response_too_large`), fewer fields or more text (`extraction_*`), a new research call (`job_not_found`) | `not_found`, `blocked_by_site`, `empty_page`, `response_too_large`, `extraction_failed`, `extraction_truncated`, `job_not_found` |
| Do not retry the same input (security policy) | `ssrf_blocked`, `egress_blocked`, `disallowed_url`, `private_url`, `disallowed_protocol` |
| Stop and tell the user (account or configuration) | `missing_api_key`, `auth_failed`, `insufficient_balance`, `budget_exceeded` (HTTP status 402) |

## FAQ

### How do I add web search to Claude Code?

Run the `claude mcp add` command in [Claude Code](#claude-code). You need a DeepSeek API key and `uv`.

### Is there a cheaper alternative to Tavily for MCP?

For retrieval, yes: see [Comparison with Tavily](#comparison-with-tavily) for the prices, sources and caveats.

### How does deepseek_research handle long-running research?

It runs as a background job and returns the report inline when it is ready within 45 seconds; otherwise the agent polls `deepseek_check_research`. See [Long-running research](#long-running-research).

### Can my agent write the research report with its own model?

Yes: `deepseek_research(..., synthesis="client")` returns the ranked sources and your agent writes the report; the server then verifies its citations and numbers. See [Client-side synthesis](#client-side-synthesis).

### Does it work with Cursor, Cline and Windsurf?

Yes, with any stdio or streamable-HTTP MCP client; see [Install in your MCP client](#install-in-your-mcp-client).

### How much does a search cost?

One DeepSeek request per search or fetch, whatever the number of results; see [Costs](#costs). Every response reports `cost_usd`.

### Can it read PDFs and JavaScript-rendered pages?

Pages are rendered by DeepSeek's service, not a local browser, so it depends on that service. A 404 on a PDF or Reddit URL falls back to the search index, and an empty page is reported as such.

### Can I extract structured data from a web page?

Yes, with `deepseek_extract` or `deepseek_extract_many`: every value needs a verbatim evidence quote from the page or it is returned as `null`.

### Is my data sent anywhere?

Only to the DeepSeek API, which does the work; see [Security](#security).

### How do I limit spending?

Set `DEEPSEEK_DAILY_BUDGET_USD`; see [Configuration](#configuration).

### Can the server run remotely, for example in Docker or n8n?

Yes, over [streamable HTTP](#streamable-http) with a bearer token, also from the [Docker](#docker) image.

### Is this an official DeepSeek project?

No. It is an independent, community-maintained server that uses your own DeepSeek API key.

## Limitations

- Fetch latency is DeepSeek's page render: an already rendered page returns in about 0.5 s, others take 3-10 s, and a missing page still takes 3-5 s before DeepSeek reports it.
- Direct fetch does not render raw binary files such as PDFs natively; see the PDF fallback above.
- Fetch returns title and content only: no final URL, status code or page metadata such as the publication date (research and search results derive a date from the URL, the index or the text).
- Search and fetch only: no site mapping or crawling, and a single provider (DeepSeek). Native MCP Tasks are not used; long research uses `deepseek_check_research`.
- The cache, rate limiter and research job store are per process; several server processes multiply the effective rate, and a research job (including a client-synthesis job waiting for its report) is only known to the process that started it.
- Local rate limits are conservative defaults and do not replace DeepSeek's own quotas.
- The HTTP transport is plain HTTP, stateless and JSON-only; there is no OAuth and no SSE stream.

## Development

```bash
git clone https://github.com/radoxtech/deepseek-search-everywhere-mcp.git
cd deepseek-search-everywhere-mcp
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test,lint,http]" build twine

pytest -q                       # offline suite; no network and no key needed
ruff check . && ruff format --check .
python -m build && twine check dist/*
python scripts/validate_server_json.py
```

Live tests (`pytest -m live`, plus the `live_llm` and `live_e2e` suites) call the real API with your own key and are opt-in. See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and commit conventions and [AGENTS.md](AGENTS.md) for the code layout and project invariants. Releases are automated with [release-please](https://github.com/googleapis/release-please) from [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/), and `server.json` is published to the [MCP Registry](https://modelcontextprotocol.io/registry) by the release workflow; the `mcp-name` comment near the top of this file is used by the registry to verify package ownership.

Release notes: [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE). Not affiliated with DeepSeek.
