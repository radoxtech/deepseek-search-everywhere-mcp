#!/usr/bin/env bash
# Smoke-test the Docker image: build, CLI checks, non-root user verification, and HTTP transport ping.
set -euo pipefail

IMAGE_NAME="deepseek-search-everywhere-mcp:smoke-test"
CONTAINER_NAME="deepseek-mcp-smoke-$RANDOM"
TEST_TOKEN="smoke-token-secret-12345"

echo "==> Building Docker image: ${IMAGE_NAME}..."
docker build -t "${IMAGE_NAME}" .

echo "==> Testing --help..."
docker run --rm "${IMAGE_NAME}" --help | grep -q "MCP server for the DeepSeek Search & Fetch API"

echo "==> Testing --version..."
docker run --rm "${IMAGE_NAME}" --version

echo "==> Verifying non-root execution..."
CONTAINER_UID=$(docker run --rm --entrypoint id "${IMAGE_NAME}" -u)
if [ "${CONTAINER_UID}" = "0" ]; then
    echo "ERROR: Container is running as root (UID 0)!" >&2
    exit 1
fi
echo "Verified running as unprivileged UID ${CONTAINER_UID}."

echo "==> Testing streamable-http server in container..."
docker run -d --name "${CONTAINER_NAME}" -p 8000:8000 \
    -e MCP_BEARER_TOKEN="${TEST_TOKEN}" \
    "${IMAGE_NAME}" --transport streamable-http --host 0.0.0.0 --port 8000

trap 'docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true' EXIT

echo "==> Waiting for HTTP server to become responsive..."
for i in $(seq 1 15); do
    if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/mcp | grep -q "[234]"; then
        break
    fi
    sleep 1
done

docker logs "${CONTAINER_NAME}"

echo "==> Querying tools/list over HTTP..."
RESPONSE=$(curl -fsS -X POST http://127.0.0.1:8000/mcp \
    -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    -H "Authorization: Bearer ${TEST_TOKEN}" \
    -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}')

echo "${RESPONSE}" | grep -q "deepseek_search"
echo "==> Docker smoke tests passed successfully!"
