"""Make the lab modules importable from the tests without installing them."""

from __future__ import annotations

import sys
from pathlib import Path

EVALS = Path(__file__).resolve().parents[1]
if str(EVALS) not in sys.path:
    sys.path.insert(0, str(EVALS))
