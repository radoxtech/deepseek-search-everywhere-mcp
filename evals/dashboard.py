"""Static HTML report for one evaluation iteration, meant to be archived and compared.

Reads only saved artifacts (no network, no LLM): a ``keypoints`` score directory, funnel JSON files
written by ``evals/funnel.py --json-out``, run record directories, and optionally one research debug
dump with ``stage_trace``. Writes ``report.html`` (self-contained, inline CSS, no JavaScript so the
untrusted page text and reports can never execute), plus ``summary.json`` and ``manifest.json``::

    python evals/dashboard.py build --label baseline --out evals/archive/2026-11-18-00-baseline \\
        --score evals/results/kpa-fullctx \\
        --funnel evals/results/kpa-fullctx/_funnel-top10.json:top10 \\
        --funnel evals/results/kpa-fullctx/_funnel-all.json:all \\
        --flow evals/results/fullctx-top10:top10 --flow evals/results/fullctx-all:all \\
        --pool evals/results/qlab --trace /path/to/job.json

The page is ordered for a reader who does not know the method: verdict first, then the funnel in
plain counts, then evidence, then method and glossary. Everything written into the HTML is escaped;
the page text of sources never reaches the output, only counts and short fact statements.
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from deepseek_search_everywhere_mcp.deep_research import (  # noqa: E402
    citation_ids,
    report_body_words,
)

STAGES = ("found", "selected", "context", "report", "supported")
VARIANT_HEADER = [
    "variant",
    "facts covered",
    "with a citation",
    "unsupported sentences",
    "cost",
    "time",
    "reports",
]
STEPS: tuple[dict[str, Any], ...] = (
    {
        "key": "plan",
        "title": "Plan the search",
        "stage": None,
        "produced": "subqueries",
        "unit": "queries",
        "loss": (),
    },
    {
        "key": "search",
        "title": "Search the web",
        "stage": "found",
        "produced": "pool_pages",
        "unit": "pages per topic",
        "loss": ("not found",),
    },
    {
        "key": "pick",
        "title": "Pick the pages",
        "stage": "selected",
        "produced": "sources",
        "unit": "pages sent to the writer",
        "loss": ("not selected",),
    },
    {
        "key": "fetch",
        "title": "Fill the gaps",
        "stage": None,
        "produced": None,
        "unit": "",
        "loss": (),
    },
    {
        "key": "trim",
        "title": "Trim the text",
        "stage": "context",
        "produced": "context_chars",
        "unit": "characters in the context",
        "loss": ("cut", "cut from context"),
    },
    {
        "key": "write",
        "title": "Write the report",
        "stage": "report",
        "produced": "words",
        "unit": "words in the report",
        "loss": ("not written", "partly written"),
    },
    {
        "key": "cite",
        "title": "Check the citations",
        "stage": "supported",
        "produced": "citations",
        "unit": "citations",
        "loss": ("unsupported",),
    },
)
FIX = {
    "search": "improve query planning: a fact that no search returned can never be written.",
    "pick": "hand more pages to the writer, or rank them better; the cut is dropping real facts.",
    "trim": "raise the character budget; the caps are cutting facts the writer already had.",
    "write": "the writer drops facts it was given, so the work is in the synthesis prompt.",
    "cite": "the writer cites sentences its sources do not back; tighten the citation rules.",
}
LEGEND = (
    (
        "facts alive after a step",
        "how many of the rubric's checked facts are still in play once that step has run. It is "
        "cumulative and can only fall; 100% means nothing was lost so far.",
    ),
    (
        "atom",
        "one smallest checkable fact inside a rubric key point. “320 USD per kWh” and “in 2024” are "
        "separate atoms of one point; a point counts by the share of its atoms a report states.",
    ),
    (
        "lost at a step",
        "facts whose first loss is that step. A fact is counted once, at the first step it did not "
        "survive, so the counts never double-count.",
    ),
    (
        "core coverage",
        "share of the rubric's key points the report covers, counted at atom level.",
    ),
    (
        "with a citation",
        "the fact is stated in a sentence whose cited passages back it. A fact stated but not "
        "supported counts as lost here.",
    ),
    (
        "unsupported sentences",
        "cited sentences per report whose cited passages do not back them. Lower is better.",
    ),
    (
        "unverified numbers",
        "numbers in the report that appear in no cited source; the server flags them with "
        "[unverified]. Different from unsupported sentences.",
    ),
    (
        "judge noise",
        "the judge is a model, not an oracle: differences below about 2 percentage points are noise "
        "and must not be read as progress.",
    ),
)
INK = "#0f172a"
MUTED = "#64748b"
RULE = "#e2e8f0"
ACCENT = "#2563eb"
GOOD = "#15803d"
BAD = "#b91c1c"
WARN = "#b45309"


def esc(value: Any) -> str:
    """Escape any value for HTML text content and attribute values."""
    return html.escape("" if value is None else str(value), quote=True)


def slug(label: str) -> str:
    """A safe HTML id fragment for a section label."""
    return "".join(ch if ch.isalnum() else "-" for ch in label.lower()).strip("-")


def tool_name(value: str | None) -> str:
    """The MCP tool name behind a record's tool label (``research`` -> ``deepseek_research``)."""
    if not value:
        return "unknown"
    return value if value.startswith("deepseek_") else f"deepseek_{value}"


def pct(value: Any) -> str:
    """A 0-1 share as a percentage with one decimal, or an em dash when missing."""
    return "—" if value is None else f"{float(value) * 100:.1f}%"


def points(value: float | None) -> str:
    """A difference of two shares in percentage points, signed, or an em dash."""
    if value is None:
        return "—"
    return f"{value * 100:+.1f} pts"


def num(value: Any, digits: int = 0) -> str:
    """A number with thousands separators and ``digits`` decimals, or an em dash."""
    return "—" if value is None else f"{float(value):,.{digits}f}"


def usd(value: Any) -> str:
    """US dollars with four decimals, the units the API bills in."""
    return "—" if value is None else f"${float(value):.4f}"


def split_line(missing: int, lost_here: int) -> str:
    """Say whether the missing facts were already gone before this step or fell here."""
    earlier = max(0, missing - lost_here)
    return f"missing {missing} = {earlier} earlier + {lost_here} here"


def stage_losses(average: dict[str, Any]) -> dict[str, float]:
    """Drop in surviving facts at each stage, with 100% as the baseline before the first one."""
    out: dict[str, float] = {}
    previous = 1.0
    for stage in STAGES:
        value = average.get(stage)
        if value is None:
            continue
        out[stage] = previous - float(value)
        previous = float(value)
    return out


def seconds(value: Any) -> str:
    """Seconds with one decimal, or an em dash."""
    return "—" if value is None else f"{float(value):.1f}s"


def git_sha() -> str:
    """The short sha of the checkout the report was generated from."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load(path: Path) -> dict[str, Any]:
    """Read a JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def table(headers: list[str], rows: list[list[str]]) -> str:
    """An HTML table; cell content is already escaped by the caller."""
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def chip(kind: str) -> str:
    """A small status label with a word, so colour is never the only signal."""
    words = {
        "bad": "loses facts here",
        "warn": "small loss here",
        "ok": "no loss here",
        "na": "not measured",
    }
    return f'<span class="chip {kind}">{words[kind]}</span>'


def stat(value: str, label: str, note: str = "", fill: float | None = None, small: bool = False) -> str:
    """One headline number with its label, optionally a bar showing it against the target."""
    bar = ""
    if fill is not None:
        width = max(0.0, min(1.0, float(fill))) * 100
        bar = f'<div class="minibar"><div style="width:{width:.1f}%"></div></div>'
    klass = "v small" if small else "v"
    return (
        f'<div class="card"><div class="{klass}">{value}</div>'
        f'<div class="k">{esc(label)}</div>{bar}<div class="n">{esc(note)}</div></div>'
    )


def verdict_sentence(funnel: dict[str, Any], values: dict[str, Any], score: dict[str, Any] | None) -> str:
    """One plain sentence: what the tool achieves, and where the biggest loss is."""
    avg = funnel.get("average") or {}
    total = int(funnel.get("atoms_total") or 0)
    kept = avg.get("report")
    supported = avg.get("supported")
    tool = tool_name(values.get("tool")) if values else "the tool"
    losses = stage_losses(avg)
    worst, worst_loss = max(losses.items(), key=lambda kv: kv[1]) if losses else (None, 0.0)
    parts = [
        f"{esc(tool)} states {esc(pct(kept))} of the important facts in its report, "
        f"{esc(pct(supported))} with a citation that backs them."
    ]
    if worst and total:
        lost = int(round(worst_loss * total))
        name = next(step["title"] for step in STEPS if step["stage"] == worst)
        parts.append(f"The biggest loss is at “{esc(name)}”: {lost} of {total} facts stop there.")
    if score:
        sides = list((score.get("sides") or {}).values())
        deltas = score.get("delta_vs_reference") or {}
        moving = next((v for k, v in deltas.items() if abs(float(v.get("core") or 0)) > 0), None)
        if len(sides) > 1 and moving:
            delta = float(moving.get("core") or 0.0)
            cost_ref = float(sides[0].get("cost_offpeak_usd") or 0.0)
            cost_new = float(sides[1].get("cost_offpeak_usd") or 0.0)
            ratio = cost_new / cost_ref if cost_ref else 0.0
            if abs(delta) < 0.02:
                parts.append(
                    f"The second variant differs by {esc(points(delta))}, which is inside judge noise."
                )
            else:
                parts.append(
                    f"The second variant moves coverage by {esc(points(delta))} at {ratio:.1f}× the cost."
                )
    return " ".join(parts)


def funnel_html(funnel: dict[str, Any], flow: dict[str, Any] | None, label: str) -> str:
    """The funnel in plain counts: one bar per step, with what it produced and what it lost."""
    avg = funnel.get("average") or {}
    total = int(funnel.get("atoms_total") or 0)
    detail = funnel.get("loss_detail") or []
    by_stage: dict[str, list[dict[str, str]]] = {}
    for item in detail:
        by_stage.setdefault(item.get("stage") or "", []).append(item)
    losses = stage_losses(avg)
    rows = []
    for stage in STAGES:
        value = avg.get(stage)
        loss = losses.get(stage, 0.0)
        step = next(s for s in STEPS if s["stage"] == stage)
        produced = ""
        if flow and step.get("produced"):
            produced = f"{esc(num(flow.get(step['produced']), 0))} {esc(step['unit'])}"
        facts = [f for lost_at in step["loss"] for f in by_stage.get(lost_at, [])]
        kind = "na" if value is None else "ok" if loss < 0.005 else "warn" if loss < 0.05 else "bad"
        lost_note = ""
        if facts:
            items = "".join(
                f"<li><code>{esc(f['topic'])} {esc(f['point'])}</code> — {esc(f['fact'])}</li>"
                for f in facts[:40]
            )
            more = f"<li>and {len(facts) - 40} more</li>" if len(facts) > 40 else ""
            noun = "fact" if len(facts) == 1 else "facts"
            lost_note = (
                f'<details class="lost"><summary>{len(facts)} {noun} lost here</summary>'
                f"<ul>{items}{more}</ul></details>"
            )
        elif loss >= 0.005:
            lost_note = '<span class="muted">details not recorded for this run</span>'
        width = max(0.0, min(1.0, float(value or 0))) * 100
        alive = int(round(float(value or 0) * total)) if total else None
        missing_total = total - alive if total and alive is not None else 0
        lost_here = int(round(loss * total)) if total else 0
        rows.append(
            f'<div class="frow"><div class="ftitle">{esc(step["title"])}</div>'
            f'<div class="fbar" title="kept {alive} of {total}">'
            f'<div class="fkept" style="width:{width:.1f}%">'
            f"{esc(str(alive)) if alive is not None else '—'} / {esc(str(total))}</div>"
            f'<div class="fmiss"></div></div>'
            f'<div class="fmeta">{esc(pct(value))}<br><span class="muted">{produced or "—"}</span></div>'
            f'<div class="floss">{chip(kind)}'
            f'<span class="muted">{esc("step cost 0 pts" if loss < 0.0005 else points(-loss))}</span>'
            f'<div class="split">{esc(split_line(missing_total, lost_here))}</div>'
            f"{lost_note}</div></div>"
        )
    return (
        f'<section id="funnel"><h2>Where the facts get lost — {esc(label)}</h2>'
        f'<p class="sub">Each row is one step of the job. The whole bar is all {esc(str(total))} '
        "checked facts: the blue part (with the “kept / total” count) still survives that step, the "
        "striped part is already lost. The right column separates what was already missing before this "
        "step from what fell <em>at</em> this step, so a row can say “no loss here” and still be far "
        "from 137.</p>"
        f'<div class="funnel">{"".join(rows)}</div></section>'
    )


def variants_html(score: dict[str, Any]) -> str:
    """Two variants side by side, as three cards plus a compact table."""
    sides = score.get("sides") or {}
    deltas = score.get("delta_vs_reference") or {}
    if len(sides) < 2:
        return ""
    names = list(sides)
    ref, other = sides[names[0]], sides[names[1]]
    delta = float((deltas.get(names[1]) or {}).get("core") or 0.0)
    noise = abs(delta) < 0.02
    cards = (
        stat(
            f"{pct(ref.get('core'))} → {pct(other.get('core'))}",
            "facts covered",
            f"{points(delta)} (noise is ±2 pts)" if not noise else f"{points(delta)} — within noise",
        )
        + stat(
            f"{usd(ref.get('cost_offpeak_usd'))} → {usd(other.get('cost_offpeak_usd'))}",
            "cost per report",
            f"{seconds(ref.get('wall_s'))} → {seconds(other.get('wall_s'))} per report",
            small=True,
        )
        + stat(
            f"{num(ref.get('unsupported'), 2)} → {num(other.get('unsupported'), 2)}",
            "unsupported sentences",
            "fewer is better",
        )
    )
    rows = [
        [
            esc(name),
            f"<code>{esc(pct(data.get('core')))}</code>",
            f"<code>{esc(pct(data.get('core_supported')))}</code>",
            f"<code>{esc(num(data.get('unsupported'), 2))}</code>",
            f"<code>{esc(usd(data.get('cost_offpeak_usd')))}</code>",
            f"<code>{esc(seconds(data.get('wall_s')))}</code>",
            f"<code>{esc(data.get('reports', '—'))}</code>",
        ]
        for name, data in sides.items()
    ]
    topic_rows: list[list[str]] = []
    for topic in sorted({t for data in sides.values() for t in (data.get("by_topic") or {})}):
        row = [esc(topic)]
        for data in sides.values():
            row.append(f"<code>{esc(pct((data.get('by_topic') or {}).get(topic, {}).get('core')))}</code>")
        topic_rows.append(row)
    return (
        '<section id="variants"><h2>Does the larger context pay off?</h2>'
        f'<p class="sub">compared with “{esc(names[0])}”</p>'
        f'<div class="cards">{cards}</div>'
        f"{table(VARIANT_HEADER, rows)}"
        f"<h3>Per topic</h3>{table(['topic', *map(esc, names)], topic_rows)}</section>"
    )


def details_html(
    funnels: list[tuple[str, dict[str, Any]]],
    flows: list[tuple[str, dict[str, Any]]],
    dump: dict[str, Any] | None,
    score: dict[str, Any] | None,
) -> str:
    """Everything a careful reader may want to check, folded away by default."""
    blocks = []
    for label, funnel in funnels:
        rows = [
            [esc(topic)] + [f"<code>{esc(pct(stages.get(stage)))}</code>" for stage in STAGES]
            for topic, stages in sorted((funnel.get("by_topic") or {}).items())
        ]
        blocks.append(f"<h3>Per topic — {esc(label)}</h3>{table(['topic', *STAGES], rows)}")
    for label, values in flows:
        if not values:
            continue
        rows = [
            ["queries per job", f"<code>{esc(num(values.get('subqueries'), 1))}</code>"],
            ["pages in the pool, per topic", f"<code>{esc(num(values.get('pool_pages')))}</code>"],
            ["pages sent to the writer", f"<code>{esc(num(values.get('sources'), 1))}</code>"],
            ["characters in the context", f"<code>{esc(num(values.get('context_chars')))}</code>"],
            ["words in the report", f"<code>{esc(num(values.get('words')))}</code>"],
            ["citations per report", f"<code>{esc(num(values.get('citations'), 1))}</code>"],
            [
                "unverified numbers per report",
                f"<code>{esc(num(values.get('unverified_numbers'), 2))}</code>",
            ],
            [
                "reports / topics",
                f"<code>{esc(values.get('reports', 0))} / {esc(values.get('topics', 0))}</code>",
            ],
            ["reports that failed", f"<code>{esc(values.get('failures', 0))}</code>"],
        ]
        blocks.append(f"<h3>Volumes — {esc(label)}</h3>{table(['measured', 'mean per report'], rows)}")
    if dump and dump.get("stage_trace"):
        rows = []
        for entry in dump["stage_trace"]:
            counters = entry.get("counters") or {}
            shown = ", ".join(f"{k}={v}" for k, v in counters.items())
            rows.append(
                [
                    esc(entry.get("stage")),
                    f"<code>{esc(entry.get('duration_ms'))} ms</code>",
                    f'<span class="mono">{esc(shown)}</span>',
                ]
            )
        blocks.append(
            f"<h3>Stage trace of one job</h3>"
            f'<p class="sub">job {esc(str(dump.get("job_id"))[:8])} · {esc(dump.get("topic"))}</p>'
            f"{table(['stage', 'duration', 'counters'], rows)}"
        )
    if score and len(score.get("sides") or {}) > 1:
        sides = score["sides"]
        rows = [
            [esc(topic)]
            + [
                f"<code>{esc(pct((d.get('by_topic') or {}).get(topic, {}).get('core')))}</code>"
                for d in sides.values()
            ]
            for topic in sorted({t for d in sides.values() for t in (d.get("by_topic") or {})})
        ]
        blocks.append(f"<h3>Coverage per topic</h3>{table(['topic', *map(esc, sides)], rows)}")
    if not blocks:
        return ""
    return (
        '<section id="details"><h2>Numbers behind the charts</h2>'
        '<p class="sub">Everything above is derived from these values; open what you want to check.</p>'
        "<details><summary>Open the tables: per topic, volumes per report, stage trace</summary>"
        f"{''.join(blocks)}</details></section>"
    )


def method_html() -> str:
    """Five sentences a junior needs before trusting the numbers."""
    return (
        '<section id="method"><h2>How this is measured</h2><ul class="method">'
        "<li>Each topic has a rubric: a handful of key points, each split into the smallest checkable "
        "facts (atoms).</li>"
        "<li>A judge model reads one report at a time and answers, per atom, whether the report states "
        "it and whether the cited passage backs it. It never sees the other variants.</li>"
        "<li>Separately, the same kind of judge labels each fetched page with the atoms it states. That "
        "is what makes “lost before reading” measurable.</li>"
        "<li>This script does no judging: it joins the verdicts with the page labels and reports where "
        "each atom disappeared first.</li>"
        "<li>Coverage is not usefulness: a report can score 50% and still be the best answer for a "
        "reader. Differences under about 2 points are judge noise.</li>"
        "</ul></section>"
    )


def legend_html() -> str:
    """The glossary, last, for readers who want the exact definitions."""
    rows = [[f"<code>{esc(term)}</code>", esc(text)] for term, text in LEGEND]
    return f'<section id="glossary"><h2>Glossary</h2>{table(["term", "meaning"], rows)}</section>'


def flow_values(run_dir: Path, pool_dir: Path | None) -> dict[str, Any]:
    """Aggregate one run directory into the volumes the report shows."""
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(run_dir.glob("*__*.json"))]
    if not records:
        return {}
    envs = [record.get("envelope") or {} for record in records]
    reports = [env.get("report") or "" for env in envs]
    dumps = [record.get("dump") or {} for record in records]

    def mean(values: Any) -> float | None:
        kept = [float(v) for v in values if v is not None]
        return statistics.fmean(kept) if kept else None

    pool = None
    if pool_dir is not None:
        sizes = [
            len(json.loads(p.read_text(encoding="utf-8"))) for p in sorted(pool_dir.glob("*/full_pages.json"))
        ]
        pool = mean(sizes)
    return {
        "tool": next((tool_name(r.get("tool")) for r in records if r.get("tool")), None),
        "reports": len(records),
        "topics": len({record.get("topic_id") for record in records}),
        "failures": sum(1 for env in envs if not env.get("ok") or not env.get("report")),
        "subqueries": mean(len(env.get("subqueries") or []) for env in envs),
        "pool_pages": pool,
        "sources": mean(len(env.get("sources") or []) for env in envs),
        "context_chars": mean(
            sum(len(s.get("content") or "") for s in dump.get("synthesis_context") or []) for dump in dumps
        ),
        "words": mean(report_body_words(report) for report in reports if report),
        "citations": mean(len(citation_ids(report)) for report in reports if report),
        "unverified_numbers": mean(env.get("unverified_number_count") or 0 for env in envs),
        "cost": mean(env.get("cost_usd") or 0.0 for env in envs),
        "wall": mean(record.get("wall_s") for record in records),
    }


def build(args: argparse.Namespace) -> None:
    """Collect the artifacts, write summary.json and manifest.json, then render report.html."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    funnels = []
    for spec in args.funnel or []:
        path, _, label = spec.partition(":")
        funnels.append((label or Path(path).parent.name, load(Path(path))))
    score = load(Path(args.score) / "_score.json") if args.score else None
    dump = load(Path(args.trace)) if args.trace else None
    pool_dir = Path(args.pool) if args.pool else None
    flows = []
    for spec in args.flow or []:
        path, _, label = spec.partition(":")
        flows.append((label or Path(path).name, flow_values(Path(path), pool_dir)))
    tools = sorted(
        {v["tool"] for _, v in flows if v.get("tool")} | ({tool_name(args.tool)} if args.tool else set())
    )
    primary_label, primary_funnel = funnels[0] if funnels else ("", {})
    primary_flow = next((values for _, values in flows if values), {})

    avg = primary_funnel.get("average") or {}
    total = int(primary_funnel.get("atoms_total") or 0)
    kept_report = int(round((avg.get("report") or 0) * total))
    kept_cited = int(round((avg.get("supported") or 0) * total))
    cards = (
        stat(
            pct(avg.get("report")),
            "facts in the report",
            f"{kept_report} of {total} · {total - kept_report} short of the target",
            fill=avg.get("report"),
        )
        + stat(
            pct(avg.get("supported")),
            "facts backed by a citation",
            f"{kept_cited} of {total} · the honest number",
            fill=avg.get("supported"),
        )
        + stat(
            usd(primary_flow.get("cost")) if primary_flow else "—",
            "cost per report",
            f"{seconds(primary_flow.get('wall'))} per report" if primary_flow else "",
        )
    )
    losses = stage_losses(avg)
    worst = max(losses.items(), key=lambda kv: kv[1]) if losses else None
    advice = ""
    if worst:
        key = next(step["key"] for step in STEPS if step["stage"] == worst[0])
        lost = int(round(worst[1] * total))
        advice = (
            f'<p class="advice"><strong>What to fix next:</strong> {esc(FIX.get(key, ""))} '
            f"({lost} of {total} facts stop there.)</p>"
        )

    summary = {
        "label": args.label,
        "iteration": args.iteration,
        "tools": tools,
        "score": ({"reference": score.get("reference"), "sides": score.get("sides")} if score else None),
        "funnels": {label: data for label, data in funnels},
        "flows": {label: values for label, values in flows},
        "trace": [e.get("stage") for e in (dump or {}).get("stage_trace", [])],
    }
    manifest = {
        "label": args.label,
        "iteration": args.iteration,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": git_sha(),
        "tools": tools,
        "artifacts": {
            "score": args.score,
            "flows": args.flow or [],
            "pool": args.pool,
            "funnels": args.funnel or [],
            "trace": args.trace,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    meta = " · ".join(
        part
        for part in (
            f"evaluation {args.iteration}",
            esc(manifest["created_at"][:16]),
            f"sha {esc(manifest['git_sha'])}",
            f"{esc(primary_flow.get('reports', ''))} reports on {esc(primary_flow.get('topics', ''))} topics"
            if primary_flow
            else "",
        )
        if part
    )
    sections = [
        f'<section id="verdict"><h2>The verdict</h2>'
        f'<p class="verdict">{verdict_sentence(primary_funnel, primary_flow, score)}</p>'
        f'<div class="cards">{cards}</div>{advice}</section>',
        funnel_html(primary_funnel, primary_flow, primary_label) if primary_funnel else "",
        variants_html(score) if score else "",
        details_html(funnels, flows, dump, score),
        method_html(),
        legend_html(),
    ]
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(tools[0] if tools else "Evaluation")} — evaluation {args.iteration}</title>
<style>
:root {{ color-scheme: light }}
* {{ box-sizing: border-box }}
body {{ margin: 0; padding: 48px 24px 72px; background: #f8fafc; color: {INK};
  font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif }}
main {{ max-width: 960px; margin: 0 auto }}
h1 {{ font-size: 30px; margin: 0 0 6px; letter-spacing: -.01em }}
h2 {{ font-size: 19px; margin: 44px 0 6px }}
h3 {{ font-size: 15px; margin: 26px 0 8px; color: {MUTED}; font-weight: 600 }}
.sub {{ color: {MUTED}; font-size: 13.5px; margin: 0 0 16px }}
.verdict {{ font-size: 21px; line-height: 1.5; margin: 8px 0 20px; max-width: 46em }}
.advice {{ background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px;
  padding: 12px 16px; margin: 18px 0 0; font-size: 15px }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px;
  margin: 18px 0 6px }}
.card {{ background: #fff; border: 1px solid {RULE}; border-radius: 12px; padding: 16px 18px }}
.card .v.small {{ font-size: 21px }}
.card .v {{ font-size: 30px; font-weight: 650; letter-spacing: -.02em;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace }}
.card .k {{ font-size: 13px; color: {MUTED}; margin-top: 2px }}
.card .n {{ font-size: 12.5px; color: {MUTED}; margin-top: 6px }}
.funnel {{ margin: 14px 0 8px; display: grid; gap: 10px }}
.frow {{ display: grid; grid-template-columns: 175px 1fr 122px 215px; gap: 14px; align-items: center }}
.ftitle {{ font-size: 14.5px; font-weight: 550 }}
.fbar {{ display: flex; height: 38px; border-radius: 8px; overflow: hidden;
  border: 1px solid #fca5a5; background: #fee2e2 }}
.fkept {{ flex: 0 0 auto; display: flex; align-items: center; padding-left: 11px; color: #fff;
  font-weight: 700; font-size: 15px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  background: #1d4ed8; white-space: nowrap }}
.fmiss {{ flex: 1 1 auto; min-width: 0;
  background: repeating-linear-gradient(45deg, #fee2e2 0 7px, #fca5a5 7px 14px) }}
.minibar {{ height: 7px; border-radius: 4px; background: #e2e8f0; margin: 12px 0 8px;
  overflow: hidden }}
.minibar div {{ height: 7px; background: #1d4ed8 }}
.fmeta {{ font-size: 14px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace }}
.floss {{ font-size: 13px; white-space: nowrap }}
.split {{ font-size: 12px; color: {MUTED}; margin-top: 3px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace }}
.chip {{ display: inline-block; font-size: 11.5px; font-weight: 650; text-transform: uppercase;
  letter-spacing: .04em; padding: 2px 8px; border-radius: 999px }}
.chip.bad {{ background: #fee2e2; color: {BAD} }}
.chip.warn {{ background: #fef3c7; color: {WARN} }}
.chip.ok {{ background: #dcfce7; color: {GOOD} }}
.chip.na {{ background: #e2e8f0; color: {MUTED} }}
.muted {{ color: {MUTED} }}
details {{ margin-top: 4px }}
summary {{ cursor: pointer; color: {MUTED}; font-size: 12.5px }}
details.lost ul {{ margin: 6px 0 0; padding-left: 18px; font-size: 12.5px; color: #334155 }}
details.lost li {{ margin: 2px 0 }}
table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; margin: 10px 0 4px }}
th {{ text-align: left; font-weight: 600; color: {MUTED}; padding: 8px 10px;
  border-bottom: 1px solid {RULE}; font-size: 12px; text-transform: uppercase; letter-spacing: .03em }}
td {{ padding: 7px 10px; border-bottom: 1px solid #f1f5f9; vertical-align: top }}
tbody tr:hover {{ background: #f8fafc }}
code, .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px }}
ul.method {{ margin: 8px 0 0; padding-left: 20px; font-size: 14.5px; color: #334155 }}
ul.method li {{ margin: 6px 0 }}
footer {{ margin-top: 56px; color: {MUTED}; font-size: 12px }}
</style></head>
<body><main>
<h1>{esc(tools[0] if tools else "Evaluation")}</h1>
<p class="sub">{esc(args.label)} · {meta}</p>
{"".join(sections)}
<footer>Generated by evals/dashboard.py · no JavaScript, no external requests ·
numbers come from saved artifacts only</footer>
</main></body></html>
"""
    (out / "report.html").write_text(page, encoding="utf-8")
    print(f"wrote {out / 'report.html'} ({len(page)} bytes)")
    print(f"wrote {out / 'summary.json'} and {out / 'manifest.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--label", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--iteration", type=int, default=0)
    b.add_argument("--score", help="result directory holding a keypoints _score.json")
    b.add_argument(
        "--funnel",
        action="append",
        help="funnel JSON from evals/funnel.py --json-out, optionally suffixed with :label",
    )
    b.add_argument(
        "--flow",
        action="append",
        help="run record directory (topic__repeat.json files), optionally suffixed with :label",
    )
    b.add_argument("--pool", help="query-lab directory with <topic>/full_pages.json for the pool size")
    b.add_argument("--trace", help="research debug dump with stage_trace")
    b.add_argument("--tool", help="MCP tool under test when the records carry no tool field")
    args = parser.parse_args()
    if args.cmd == "build":
        build(args)


if __name__ == "__main__":
    main()
