"""Make the src-layout package importable when it is not installed."""

import importlib.util
import pathlib
import sys

if importlib.util.find_spec("deepseek_search_everywhere_mcp") is None:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
