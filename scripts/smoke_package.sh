#!/usr/bin/env bash
# Build the wheel, install it into a clean virtual environment and exercise the CLI entry point.
# Requires: python with the `build` package, and `uv`.
set -euo pipefail

cd "$(dirname "$0")/.."

PROG=deepseek-search-everywhere-mcp
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

rm -rf dist
python -m build --wheel --outdir dist >/dev/null
wheel="$(ls dist/*.whl)"
echo "Built $wheel"

uv venv --python 3.12 --quiet "$workdir/venv"
uv pip install --quiet --python "$workdir/venv/bin/python" "$wheel"

echo "--- $PROG --version"
"$workdir/venv/bin/$PROG" --version
echo "--- $PROG --help"
"$workdir/venv/bin/$PROG" --help

echo "--- uv tool run --from $wheel $PROG --version"
uv tool run --from "$wheel" "$PROG" --version

echo "Smoke test passed"
