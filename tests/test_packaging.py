"""The built wheel must ship the package, and only the package (no stray top-level ``server.py``)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("build") is None, reason="the `build` package is not installed"
)


@pytest.fixture(scope="module")
def wheel_names(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    out = tmp_path_factory.mktemp("wheel")
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out), str(ROOT)],
        check=True,
        capture_output=True,
    )
    (wheel,) = out.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        return archive.namelist()


def test_wheel_contains_the_package(wheel_names: list[str]) -> None:
    assert "deepseek_search_everywhere_mcp/__init__.py" in wheel_names
    assert "deepseek_search_everywhere_mcp/cli.py" in wheel_names


def test_wheel_has_no_top_level_server_module(wheel_names: list[str]) -> None:
    assert "server.py" not in wheel_names
    top_level = {name.split("/")[0] for name in wheel_names}
    assert {name for name in top_level if not name.endswith(".dist-info")} == {"deepseek_search_everywhere_mcp"}
