# Security Policy

This project is an independent, community-maintained MCP server. It is **not affiliated with DeepSeek**.

## Supported versions

| Version | Supported |
| ------- | --------- |
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

## Data sent to DeepSeek

Tool inputs (queries, URLs, and request options) are sent to the DeepSeek Search & Fetch API to perform the work.
Results returned by DeepSeek are passed back to the MCP client. The server has no other outbound destinations.

## SSRF scope

URL validation in this server is **static validation only**: it checks the URL scheme and host syntax before a request is
made. The server does not resolve DNS locally. **DeepSeek performs the actual fetch**, so the SSRF protection
offered by this server is limited to rejecting unsafe URL forms before they are forwarded (internal hostnames,
literal private or loopback addresses, and URLs that embed credentials such as `https://user:pass@host`; the latter
is a behavior change in v0.2 and returns the `ssrf_blocked` error code). It does not guarantee
network-level isolation of the DeepSeek fetch infrastructure.

## Telemetry

This server has no telemetry. It does not phone home and does not collect usage data.
