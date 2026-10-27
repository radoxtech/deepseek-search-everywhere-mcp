"""The package version must be single-sourced across pyproject, server.json and ``__version__``."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import deepseek_search_everywhere_mcp

ROOT = Path(__file__).resolve().parent.parent


def test_versions_are_consistent() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    expected = pyproject["project"]["version"]
    assert deepseek_search_everywhere_mcp.__version__ == expected
    assert manifest["version"] == expected
    assert [pkg["version"] for pkg in manifest["packages"]] == [expected]


def test_manifest_is_stdio_only_and_declares_secret_key() -> None:
    manifest = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    assert "remotes" not in manifest
    (package,) = manifest["packages"]
    assert package["transport"] == {"type": "stdio"}
    env = {var["name"]: var for var in package["environmentVariables"]}
    assert env["DEEPSEEK_API_KEY"]["isRequired"] is True
    assert env["DEEPSEEK_API_KEY"]["isSecret"] is True
