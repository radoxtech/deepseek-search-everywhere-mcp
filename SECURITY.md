# Security Policy

This project is an independent, community-maintained MCP server. It is **not affiliated with DeepSeek**.

## Supported versions

| Version | Supported |
| ------- | --------- |
| 3.x     | Yes       |
| 0.2.x   | Yes       |
| < 0.2   | No        |

## Reporting a vulnerability

Please do not open a public issue for security problems.

Report vulnerabilities privately through GitHub private vulnerability reporting:
go to the repository's **Security** tab and choose **Report a vulnerability**.
Include a description, reproduction steps, and the affected version.

## API key handling

- The DeepSeek API key is read from the `DEEPSEEK_API_KEY` environment variable.
- As a legacy fallback, the key may be read from the opencode credentials file (`~/.local/share/opencode/auth.json`,
  or the path in `DEEPSEEK_AUTH_JSON`). This fallback is kept for compatibility with opencode and is documented as legacy.
- The key is sent only in the `Authorization` header to the configured DeepSeek API base URL (`DEEPSEEK_BASE_URL`).
- Do not commit `.env` files or credentials. `.env.example` is the only env file tracked in the repository.

## Data flow

The server has one outbound destination: the DeepSeek API at `DEEPSEEK_BASE_URL` (default `https://api.deepseek.com`).

| Data | Where it goes | Why |
| ---- | ------------- | --- |
| Search queries and filters | DeepSeek Search API | web search |
| URLs passed to fetch, extract and research | DeepSeek Fetch API, which downloads the page | page retrieval |
| Page text, titles and snippets selected for extraction, research synthesis, query decomposition and gap analysis | DeepSeek chat completions (`deepseek-flash`) | server-side LLM steps |
| Optional query text when `rewrite=true` | DeepSeek chat completions | query rewrite |
| API key | `Authorization` header to the configured base URL only | authentication |

Results come back to the MCP client. Nothing is sent to any other service, there is no telemetry, and the API key never
appears in a prompt, a query, a URL, a log line or an error message (secrets are redacted from errors and notes).
Because queries and URLs leave your machine, do not put secrets in them; the egress guard below blocks the most common
forms. The local state is limited to in-memory caches and, if `DEEPSEEK_DAILY_BUDGET_USD` is set, a small budget file in
`DEEPSEEK_STATE_DIR`.

## Prompt injection and untrusted content

### Threat model

Any web page, search snippet or title the server retrieves is attacker-controlled. There are two victims:

1. **The server's own LLM calls** (query decomposition, gap analysis, synthesis, extraction, query rewrite).
2. **The consuming agent** (Claude Code, Cursor and others), which reads tool output and may have shell, file or other
   MCP tools. A page can try to make that agent exfiltrate data, for example by asking it to fetch a URL that contains a
   secret or to run a command.

### Controls

- **Server-side LLM calls have no tools and no secrets.** They never use tool calling, so injected text cannot trigger an
  action, and prompts contain only the task, the user's topic or schema and the retrieved text. Every chat payload sets
  the reasoning mode explicitly.
- **Sanitization of untrusted text.** Unicode tag characters, bidirectional controls and zero-width characters (which hide
  instructions from humans but not from models) are stripped from content, snippets, summaries, titles, reports and
  extracted values; zero-width joiners inside emoji sequences are kept. Markdown images are rewritten to inert
  `[image: alt](url)` text and links whose target is not http or https are dropped, so a rendering client cannot be made
  to load an attacker URL automatically.
- **Synthesis hardening.** Research sources are wrapped in delimiters containing a per-call random nonce, delimiter-like
  strings inside sources are neutralized, and the prompt declares the sources to be passive data that must never be obeyed.
- **Output filter.** In the research report, markdown images and raw HTML are removed, links that are not in the returned
  source list are reduced to their text, and sentences addressed to an AI agent (for example "ignore previous
  instructions", tool names, "you must call") are dropped and counted in `injection_suspected`. Citations are
  normalized to `[n]` and ids outside the source list are removed.
- **Extraction integrity.** Each extracted value must come with a verbatim quote that is found in the page text;
  unverifiable values are returned as `null` with a note.
- **Egress guard.** Outgoing search queries and fetch URLs are checked for secret-like data (`sk-` keys, bearer tokens,
  JWTs, AWS access keys, `api_key=`, `token=`, `secret=` and `password=` pairs, URL credentials), case-insensitively, and
  rejected with `egress_blocked`. `DEEPSEEK_FETCH_ALLOW_DOMAINS` and `DEEPSEEK_FETCH_DENY_DOMAINS` restrict fetch targets,
  and `DEEPSEEK_EGRESS_ENTROPY_CHECK` additionally rejects long high-entropy tokens.
- **Source steering.** Queries generated by the gap-analysis round are validated (no URLs, no unexpected `site:`), and a
  per-domain cap and a host denylist (`DEEPSEEK_SOURCE_DENYLIST`) limit how much one site can dominate a report.
- **Provenance flags.** Responses carrying web content set `untrusted_content: true`; research and extraction output,
  which is model-written from untrusted pages, also sets `derived_from_untrusted: true`.
- **Cost bounds.** Per-tier token limits, a per-call cost ceiling and the optional daily budget limit the damage of
  content designed to induce very long outputs.

### Residual risks

- Filters reduce risk, they do not remove it. A page can still contain misleading claims that a model repeats in a report
  or an extracted value that matches a real but poisoned quote.
- **The consuming agent must still treat all returned content as untrusted data.** Do not let an agent follow
  instructions found in tool output, and do not give it tools that can act on them without confirmation.
- The egress guard recognizes known secret formats; an unusual secret embedded in a query or URL can pass.
- Static SSRF checks do not resolve DNS (see below), so a public hostname that resolves to an internal address is only
  stopped by DeepSeek's fetch infrastructure.

## SSRF scope

URL validation in this server is **static validation only**: it checks the URL scheme and host syntax before a request is
made. The server does not resolve DNS locally. **DeepSeek performs the actual fetch**, so the SSRF protection
offered by this server is limited to rejecting unsafe URL forms before they are forwarded (internal hostnames,
literal private or loopback addresses, and URLs that embed credentials such as `https://user:pass@host`; the latter
is a behavior change in v0.2 and returns the `ssrf_blocked` error code). It does not guarantee
network-level isolation of the DeepSeek fetch infrastructure.

## HTTP transport

- The streamable HTTP transport always requires a bearer token (`MCP_BEARER_TOKEN`). Requests to the MCP path
  without `Authorization: Bearer <token>`, or with a different token, get `401` before reaching the MCP server.
  Tokens are compared in constant time and are never logged.
- DNS-rebinding protection (Host/Origin checks) of the MCP SDK is disabled on purpose: the bearer token is the trust
  boundary, and the Host check would reject legitimate clients when the server binds to `0.0.0.0`.
- The default bind address is `127.0.0.1`. When exposing the server beyond loopback, use a long random token and put
  TLS in front of it (for example a reverse proxy); the server itself speaks plain HTTP.

## Telemetry

This server has no telemetry. It does not phone home and does not collect usage data.
