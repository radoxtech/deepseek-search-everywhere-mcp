"""Validate ``server.json`` against the MCP Registry requirements.

Performs an offline structural check (required fields, types, length limits) and, when the
``mcp-publisher`` CLI is on ``PATH``, additionally runs ``mcp-publisher validate``.

Usage: ``python scripts/validate_server_json.py [path/to/server.json]``
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

NAME_PATTERN = re.compile(r"^[a-zA-Z0-9.-]+/[a-zA-Z0-9._-]+$")


def _require_str(obj: dict, key: str, where: str, errors: list[str], *, max_len: int | None = None) -> None:
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        errors.append(f"{where}.{key}: required non-empty string")
    elif max_len is not None and len(value) > max_len:
        errors.append(f"{where}.{key}: longer than {max_len} characters")


def _check_env_var(var: object, where: str, errors: list[str]) -> None:
    if not isinstance(var, dict):
        errors.append(f"{where}: must be an object")
        return
    _require_str(var, "name", where, errors)
    for flag in ("isRequired", "isSecret"):
        if flag in var and not isinstance(var[flag], bool):
            errors.append(f"{where}.{flag}: must be a boolean")
    for key in ("description", "default", "format"):
        if key in var and not isinstance(var[key], str):
            errors.append(f"{where}.{key}: must be a string")


def _check_package(pkg: object, where: str, errors: list[str]) -> None:
    if not isinstance(pkg, dict):
        errors.append(f"{where}: must be an object")
        return
    _require_str(pkg, "registryType", where, errors)
    _require_str(pkg, "identifier", where, errors)
    _require_str(pkg, "version", where, errors)
    if pkg.get("version") == "latest":
        errors.append(f"{where}.version: must not be 'latest'")
    transport = pkg.get("transport")
    if not isinstance(transport, dict) or transport.get("type") != "stdio":
        errors.append(f"{where}.transport.type: must be 'stdio'")
    env_vars = pkg.get("environmentVariables", [])
    if not isinstance(env_vars, list):
        errors.append(f"{where}.environmentVariables: must be an array")
        return
    for i, var in enumerate(env_vars):
        _check_env_var(var, f"{where}.environmentVariables[{i}]", errors)


def check(data: object) -> list[str]:
    """Return a list of problems found in a parsed ``server.json`` document."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["document must be a JSON object"]
    _require_str(data, "name", "server", errors, max_len=200)
    if isinstance(data.get("name"), str) and not NAME_PATTERN.match(data["name"]):
        errors.append("server.name: must look like 'io.github.<owner>/<name>'")
    _require_str(data, "description", "server", errors, max_len=100)
    _require_str(data, "version", "server", errors, max_len=255)
    if "title" in data:
        _require_str(data, "title", "server", errors, max_len=100)
    repo = data.get("repository")
    if repo is not None:
        if not isinstance(repo, dict):
            errors.append("server.repository: must be an object")
        else:
            _require_str(repo, "url", "server.repository", errors)
            _require_str(repo, "source", "server.repository", errors)
    if "remotes" in data:
        errors.append("server.remotes: not allowed (stdio-only manifest)")
    packages = data.get("packages")
    if not isinstance(packages, list) or not packages:
        errors.append("server.packages: required non-empty array")
    else:
        for i, pkg in enumerate(packages):
            _check_package(pkg, f"packages[{i}]", errors)
    return errors


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent / "server.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read {path}: {exc}", file=sys.stderr)
        return 1
    errors = check(data)
    for message in errors:
        print(f"error: {message}", file=sys.stderr)
    if errors:
        return 1
    print(f"{path}: structural checks passed")
    publisher = shutil.which("mcp-publisher")
    if publisher is None:
        print("mcp-publisher not found; skipping registry validation")
        return 0
    return subprocess.run([publisher, "validate", str(path)], check=False).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv))
