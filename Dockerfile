# Multi-stage build for deepseek-search-everywhere-mcp
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md LICENSE server.json ./
COPY src/ src/

RUN pip install --no-cache-dir build && \
    python -m build --wheel

# Runtime stage: minimal unprivileged container
FROM python:3.12-slim AS runtime

# Create dedicated non-root user (UID 10001)
RUN groupadd -g 10001 mcpuser && \
    useradd -u 10001 -g mcpuser -s /bin/false -m mcpuser

WORKDIR /app

# Install the built package with [http] dependencies
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl[http] && \
    rm -rf /tmp/*.whl

USER 10001:10001

# Default HTTP transport port
EXPOSE 8000

# Default entrypoint runs the MCP CLI (supports stdio, HTTP, --check)
ENTRYPOINT ["deepseek-search-everywhere-mcp"]
