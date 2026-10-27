"""Live-suite wiring: restore the real key for every live test and report the judge verdicts."""

from __future__ import annotations

import json
import pathlib

import pytest
from live_judge import RECORDS

REPORT_PATH = pathlib.Path(__file__).resolve().parents[2] / ".sisyphus" / "evidence" / "live-report.json"


@pytest.fixture(autouse=True)
def _live_key(live_api_key: None) -> None:
    """Give every test in this directory the real key (or skip it when there is none)."""


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Print each judged scenario's scores and reasons, then write the JSON report."""
    if not RECORDS:
        return
    terminalreporter.section("LLM judge verdicts")
    for record in RECORDS:
        terminalreporter.write_line(
            f"{record.scenario}: mean {record.mean:.2f}, relevant {record.relevant_share:.0%} "
            f"- {record.query}"
        )
        for url, verdict in zip(record.urls, record.verdicts, strict=False):
            target = url if len(record.verdicts) == len(record.urls) else "(all sources)"
            terminalreporter.write_line(f"    [{verdict.score}] {target}: {verdict.reason}")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps([r.to_dict() for r in RECORDS], ensure_ascii=False, indent=2))
    terminalreporter.write_line("report: .sisyphus/evidence/live-report.json")
