# Server Configuration, Envelopes & Error Codes

This reference documents all environment variables, local system limits, envelope response fields, and closed error codes supported by **DeepSeek Search Everywhere MCP**.

---

## 1. Environment Variables

The server reads configuration from environment variables at startup. Variables can be exported in your shell or set inside your MCP client's `env` configuration block.

### Required Variables

| Variable | Transport | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | All | Your DeepSeek API key from [platform.deepseek.com](https://platform.deepseek.com). |
| `MCP_BEARER_TOKEN` | Streamable HTTP | Mandatory secret token clients must pass in `Authorization: Bearer <TOKEN>`. The server refuses to start in HTTP mode without it. |

### Optional Variables

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | DeepSeek API base URL. |
| `DEEPSEEK_MCP_CACHE` | `1` | Set to `0` to disable in-memory response caches. |
| `DEEPSEEK_MCP_LOG_LEVEL` | `WARNING` | Log verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `DEEPSEEK_DAILY_BUDGET_USD` | Unlimited | Maximum spend cap per UTC day in USD (e.g. `5.00`). Once reached, requests fail with `budget_exceeded`. |
| `DEEPSEEK_STATE_DIR` | `~/.cache/deepseek-search-everywhere-mcp` | Directory for the persistent budget state file (`budget.json`). |
| `DEEPSEEK_SOURCE_DENYLIST` | Unset | Comma-separated domains research tools will never use as sources. |
| `DEEPSEEK_FETCH_ALLOW_DOMAINS` | Unset | Comma-separated whitelist: `deepseek_fetch` will only fetch pages from these domains. |
| `DEEPSEEK_FETCH_DENY_DOMAINS` | Unset | Comma-separated blacklist: `deepseek_fetch` will reject URLs from these domains. |
| `DEEPSEEK_EGRESS_ENTROPY_CHECK` | `0` | Set to `1` to reject outgoing queries containing long high-entropy tokens (detects accidental API key leaks). |

---

## 2. Local Limits, Caching & Concurrency

The server enforces internal safety caps that operate independently of DeepSeek's API quotas:

| Subsystem | Parameter | Limit / Default |
|---|---|---|
| **Search Cache** | TTL / Max Entries | 5 minutes / 128 items |
| **Fetch Cache** | TTL / Max Entries | 30 minutes / 128 items |
| **Query Rewrite Cache** | TTL / Max Entries | 24 hours / 1024 items |
| **Search Rate Limiter** | Sustained / Burst | 10 requests/s sustained, burst capacity of 20 |
| **Fetch Rate Limiter** | Sustained / Burst | 5 requests/s sustained, burst capacity of 10 |
| **LLM Call Rate Limiter** | Sustained / Burst | 10 requests/s sustained, burst capacity of 20 |
| **Max Concurrent Research Jobs** | In-flight ceiling | 3 concurrent jobs per server process |
| **Inline Research Timeout** | Tool call timeout | 45 seconds (exceeding calls transition to background polling) |

- **Cache Behavior:** Only successful, non-empty responses are cached. Errors are never cached. Passing `refresh=true` on any tool bypasses the cache for that specific call.
- **Single-Flight Coalescing:** Simultaneous identical queries share a single upstream request to avoid duplicate billing.
- **Automatic Retries:** Upstream HTTP 429 and transient 5xx responses are retried once within the request deadline.

---

## 3. Response Format & Envelope Specification

All tool responses return a JSON envelope containing the `ok` boolean:

### Common Envelope Fields

| Field | Present In | Description |
|---|---|---|
| `ok` | All | `true` on success, `false` on failure. |
| `error` | Failures | Error details: `{ "code", "message", "action", "reason", "param"?, "status"?, "upstream_code"?, "upstream_message"? }`. |
| `error.action` | Failures | Recommended next step: `retry`, `fix_input`, `change_approach`, `do_not_retry`, `start_again` or `ask_user`. |
| `error.reason` | Failures | Fine-grained cause from the closed reason set in `errors.py`. |
| `error.param` | Input failures | Name of the offending input parameter, omitted when no single parameter is at fault. |
| `elapsed_ms` | All except `deepseek_check_research` (not timed) | Wall-clock execution time in milliseconds. |
| `cost_usd` | Retrieval tools | Billed cost of the call in USD (`0.0` for cache hits). |
| `cost_breakdown` | Extract & Research | Subtotal breakdown: `{ "search", "fetch", "llm" }` in USD. |
| `requests_used` | All except `deepseek_check_balance`, single-URL `deepseek_fetch`/`deepseek_extract`, and running/early-error `deepseek_check_research` | Number of upstream DeepSeek API requests consumed. |
| `cached` | All except `deepseek_check_balance` and running/early-error `deepseek_check_research` | `true` if the result was served from the local cache. |
| `notes` | Successes | Diagnostic details (deduplications, truncated terms, applied filters). |
| `partial` | Batch & Research | `true` if a subset of batch items or sources failed while the overall operation succeeded. |
| `untrusted_content` | Web content tools | Always `true` to remind agents that retrieved text is untrusted data. |
| `derived_from_untrusted` | Research | `true` when model-generated research output is synthesized from untrusted pages. |
| `injection_suspected` | Research | Count of suspect prompt injection instructions filtered from source text. |
| `unverified_numbers` | Research | List of numeric claims in cited sentences not found verbatim in sources. |
| `unverified_number_count` | Research | Total count of unverified numbers (0 indicates 100% numerical grounding). |
| `status` | Research | Current lifecycle state: `"done"`, `"running"`, or `"needs_input"` (client synthesis). |
| `job_id` | Research | UUID of the background job (used for polling with `deepseek_check_research`). |
| `progress` | Research (running) | Progress object: `{ "phase", "step", "of", "phase_elapsed_ms", "deadline_ms" }`. |
| `next_action` | Research (running) | Actionable instruction advising the agent how to poll or provide input. |

---

## 4. Closed Error Codes

The server maps all errors into a closed set of well-defined codes. Agents should handle them according to these guidelines:

### 1. Transient Errors (Retry after short backoff)
- `rate_limited_local`: Local rate limit bucket exhausted.
- `upstream_rate_limited`: DeepSeek API returned HTTP 429.
- `timeout` / `upstream_timeout`: Network or connection timeout.
- `deadline_exceeded`: Execution exceeded the maximum tier deadline.
- `network_error`: Network partition or socket failure.
- `upstream_error` / `request_failed`: Upstream 5xx gateway error.
- `invalid_json`: Unparseable response received from upstream.
- `internal_error`: Unexpected server runtime exception.

### 2. Client Input Errors (Fix arguments before retrying)
- `bad_request`: Invalid parameters or incompatible options.
- `empty_query` / `empty_url`: Required query or URL string was blank.
- `query_too_long`: Query exceeds the 128-character limit.
- `topic_too_long`: Topic exceeds the 4000-character limit.
- `bad_count`: Count outside permitted range (1–50).
- `bad_lang`: Invalid ISO language code.
- `bad_url` / `url_too_long`: Malformed URL or URL exceeds length cap.
- `bad_domain` / `bad_date` / `bad_term` / `bad_phrase` / `filters_too_long`: Invalid search filter syntax.

### 3. Content & Target Errors (Change URL or approach)
- `not_found`: Target URL returned HTTP 404.
- `blocked_by_site`: Target site blocked crawler or returned HTTP 403.
- `empty_page`: Target page contained no parseable text.
- `response_too_large`: Document exceeds maximum processing buffer.
- `extraction_failed` / `extraction_truncated`: Extraction schema could not be fulfilled.
- `job_not_found`: Polled research job expired or was purged during restart.

### 4. Security & Safety Blocks (Do NOT retry identical input)
- `ssrf_blocked`: Static filter blocked private, loopback, or non-routable address.
- `egress_blocked`: Outgoing query was intercepted by the secret/entropy egress guard.
- `disallowed_url` / `private_url` / `disallowed_protocol`: URL violates security policy.

### 5. Fatal / Administrative Errors (Notify user)
- `missing_api_key`: `DEEPSEEK_API_KEY` was not provided.
- `auth_failed`: DeepSeek API rejected the key (HTTP 401).
- `insufficient_balance`: DeepSeek account balance exhausted (HTTP 402).
- `budget_exceeded`: Daily spending cap reached (`DEEPSEEK_DAILY_BUDGET_USD`).
