# MCP Client Configuration Guide

This guide provides tested configuration snippets and setup instructions for adding **DeepSeek Search Everywhere MCP** to every major MCP client and AI code editor.

All examples use `uvx` (from [Astral uv](https://docs.astral.sh/uv/)) so nothing is installed globally. Alternatively, you can install the package once with `pip install deepseek-search-everywhere-mcp` and use `deepseek-search-everywhere-mcp` directly as the command.

Replace `sk-...` with your actual DeepSeek API key from [platform.deepseek.com](https://platform.deepseek.com).

---

## 1. Claude Code (CLI)

Add the server directly from the command line:

```bash
claude mcp add deepseek-search-everywhere -e DEEPSEEK_API_KEY=sk-... -- \
  uvx deepseek-search-everywhere-mcp
```

To verify the installation:
```bash
claude mcp list
```

---

## 2. Claude Desktop

Edit your Claude Desktop configuration file:
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux:** `~/.config/Claude/claude_desktop_config.json`

Add the server to the `mcpServers` block:

```json
{
  "mcpServers": {
    "deepseek-search-everywhere": {
      "command": "uvx",
      "args": ["deepseek-search-everywhere-mcp"],
      "env": {
        "DEEPSEEK_API_KEY": "sk-..."
      }
    }
  }
}
```

Restart Claude Desktop to load the tools.

---

## 3. Cursor

In your Cursor project root, create or edit `.cursor/mcp.json` (or edit `~/.cursor/mcp.json` for all projects):

```json
{
  "mcpServers": {
    "deepseek-search-everywhere": {
      "command": "uvx",
      "args": ["deepseek-search-everywhere-mcp"],
      "env": {
        "DEEPSEEK_API_KEY": "sk-..."
      }
    }
  }
}
```

---

## 4. Cline & Roo Code (VS Code Extension)

In VS Code, open the extension MCP settings or edit `cline_mcp_settings.json`:

```json
{
  "mcpServers": {
    "deepseek-search-everywhere": {
      "command": "uvx",
      "args": ["deepseek-search-everywhere-mcp"],
      "env": {
        "DEEPSEEK_API_KEY": "sk-..."
      },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

---

## 5. Windsurf

Edit `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "deepseek-search-everywhere": {
      "command": "uvx",
      "args": ["deepseek-search-everywhere-mcp"],
      "env": {
        "DEEPSEEK_API_KEY": "sk-..."
      }
    }
  }
}
```

---

## 6. Zed

In Zed, open your `settings.json` (via Command Palette or `Cmd+,` / `Ctrl+,`) and add under `context_servers`:

```json
{
  "context_servers": {
    "deepseek-search-everywhere": {
      "source": "custom",
      "command": "uvx",
      "args": ["deepseek-search-everywhere-mcp"],
      "env": {
        "DEEPSEEK_API_KEY": "sk-..."
      }
    }
  }
}
```

---

## 7. OpenCode

In your `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "deepseek-search-everywhere": {
      "type": "local",
      "command": ["uvx", "deepseek-search-everywhere-mcp"],
      "environment": {
        "DEEPSEEK_API_KEY": "{env:DEEPSEEK_API_KEY}"
      },
      "enabled": true
    }
  }
}
```

---

## 8. Streamable HTTP (Remote Hosting)

For remote server deployments, multi-tenant setups, or platforms that connect over HTTP rather than stdio.

### Starting the HTTP Server

Install the HTTP extra (includes `uvicorn`):

```bash
pip install "deepseek-search-everywhere-mcp[http]"
export DEEPSEEK_API_KEY=sk-...
export MCP_BEARER_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"

# Start on loopback
deepseek-search-everywhere-mcp --transport streamable-http --port 8000
```

> **Security Note:** A bearer token is mandatory. The server refuses to start without `MCP_BEARER_TOKEN`. When exposing beyond `127.0.0.1`, always place a reverse proxy with TLS (such as Caddy, Nginx, or Cloudflare Tunnel) in front of it.

### Connecting MCP Clients over HTTP

In your client's `mcpServers` configuration:

```json
{
  "mcpServers": {
    "deepseek-search-everywhere": {
      "url": "http://127.0.0.1:8000/mcp",
      "headers": {
        "Authorization": "Bearer <YOUR_MCP_BEARER_TOKEN>"
      }
    }
  }
}
```

Connecting Claude Code over HTTP:
```bash
claude mcp add --transport http deepseek-search-everywhere http://127.0.0.1:8000/mcp \
  --header "Authorization: Bearer $MCP_BEARER_TOKEN"
```

---

## 9. Docker Deployment

Container images are published on GitHub Container Registry and run as an unprivileged user (UID 10001).

```bash
docker pull ghcr.io/radoxtech/deepseek-search-everywhere-mcp:latest
```

### Stdio Mode for Desktop Clients

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

### Remote HTTP Daemon

```bash
docker run -d --name deepseek-mcp -p 8000:8000 \
  -e DEEPSEEK_API_KEY="sk-..." \
  -e MCP_BEARER_TOKEN="your-secret-token" \
  ghcr.io/radoxtech/deepseek-search-everywhere-mcp:latest \
  --transport streamable-http --host 0.0.0.0 --port 8000
```

---

## 10. Troubleshooting & Verification

### Verifying Connection & Balance from CLI
To test whether your API key is valid and the server can communicate with DeepSeek without launching an MCP client, run:

```bash
export DEEPSEEK_API_KEY=sk-...
uvx deepseek-search-everywhere-mcp --check
```
This prints your DeepSeek balance in USD and exits cleanly.

### Common Issues
- **`command not found: uvx`**: Install `uv` via `curl -LsSf https://astral.sh/uv/install.sh | sh` or use `python3 -m pip install deepseek-search-everywhere-mcp`.
- **`missing_api_key`**: Ensure `DEEPSEEK_API_KEY` is present in the `env` dictionary of your client configuration file.
- **Python version error**: Python 3.12 or newer is required.
