"""Code metrics for one research record, and a per-depth summary of a run directory.

Metrics (no LLM involved):

- ``citation_valid_rate``: share of [n] citations that point to a returned source id.
- ``citation_support_rate``: share of cited sentences that one of their cited sources supports
  lexically: at least ``SUPPORT_MIN_OVERLAP`` of the sentence's content words, and every number in the
  sentence, appear in that source's text. A proxy for faithfulness, compared between variants with the
  A/A spread as tolerance.
- ``unverified_numbers``: numbers the server could not ground in their cited source.
- ``sources`` and ``domains``: count and diversity of returned sources.
- ``cost_offpeak_usd``: web requests at their flat price plus chat tokens at DeepSeek off-peak rates.
- ``wall_s`` and ``words``.

Usage: ``python evals/metrics.py evals/results/<run>`` prints the per-depth summary as JSON.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import urllib.parse
from pathlib import Path
from typing import Any

SUPPORT_MIN_OVERLAP = 0.5

# DeepSeek off-peak rates per 1M tokens (settings.DEFAULT_LLM_PRICING.off_peak).
OFFPEAK_CACHE_HIT = 0.003
OFFPEAK_CACHE_MISS = 0.15
OFFPEAK_OUTPUT = 0.60

_CITATION = re.compile(r"\[(\d+)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"'])")
_WORD = re.compile(r"[a-z][a-z\-]{3,}")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")
_STOP = frozenset(
    "this that with from have been were which their there these those than then also into more most such "
    "only other some what when where while will would could should about after before between both each "
    "many much over under very within without across among being does including however although".split()
)


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def _numbers(text: str) -> set[str]:
    return {n.replace(",", ".") for n in _NUMBER.findall(text)}


def cited_sentences(report: str) -> list[tuple[str, list[int]]]:
    """Return ``(sentence without citations, cited ids)`` for every sentence that cites something."""
    out: list[tuple[str, list[int]]] = []
    for line in report.splitlines():
        if line.lstrip().startswith("#"):
            continue
        for sentence in _SENTENCE.split(line):
            ids = [int(i) for i in _CITATION.findall(sentence)]
            if ids:
                out.append((_SPACE_BEFORE_PUNCT.sub(r"\1", _CITATION.sub("", sentence)).strip(), ids))
    return out


def supports(sentence: str, source_text: str) -> bool:
    """True when ``source_text`` holds enough of the sentence's content words and all of its numbers."""
    words = _content_words(sentence)
    if not words:
        return True
    lowered = source_text.lower()
    overlap = sum(1 for w in words if w in lowered) / len(words)
    source_numbers = _numbers(source_text)
    return overlap >= SUPPORT_MIN_OVERLAP and _numbers(sentence) <= source_numbers


def _host(url: str) -> str:
    host = urllib.parse.urlsplit(url or "").hostname or ""
    return host.removeprefix("www.")


def offpeak_cost(record: dict[str, Any]) -> float:
    """Recompute a record's cost: flat web requests plus chat tokens at off-peak rates."""
    breakdown = (record.get("envelope") or {}).get("cost_breakdown") or {}
    web = float(breakdown.get("search", 0.0)) + float(breakdown.get("fetch", 0.0))
    llm = 0.0
    for call in record.get("chat_usage") or []:
        usage = call.get("usage") or {}
        prompt = int(usage.get("prompt_tokens", 0))
        hit = int(usage.get("prompt_cache_hit_tokens", 0))
        miss = int(usage.get("prompt_cache_miss_tokens", max(0, prompt - hit)))
        out = int(usage.get("completion_tokens", 0))
        llm += (hit * OFFPEAK_CACHE_HIT + miss * OFFPEAK_CACHE_MISS + out * OFFPEAK_OUTPUT) / 1_000_000
    return round(web + llm, 6)


def record_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """All code metrics of one record (a failed job gets ``ok: False`` and its error reason)."""
    env = record.get("envelope") or {}
    base = {
        "topic_id": record.get("topic_id"),
        "depth": record.get("depth"),
        "variant": record.get("variant"),
        "repeat": record.get("repeat"),
        "wall_s": record.get("wall_s"),
        "cost_offpeak_usd": offpeak_cost(record),
    }
    if not env.get("ok") or env.get("status") != "done":
        return {**base, "ok": False, "reason": (env.get("error") or {}).get("reason") or env.get("status")}
    report = env.get("report") or ""
    sources = env.get("sources") or []
    ids = {s.get("id") for s in sources}
    texts = {
        c.get("id"): c.get("content") or ""
        for c in ((record.get("dump") or {}).get("synthesis_context") or [])
    }
    citations = [int(i) for i in _CITATION.findall(report)]
    sentences = cited_sentences(report)
    supported = sum(
        1 for sentence, cited in sentences if any(supports(sentence, texts.get(i, "")) for i in cited)
    )
    return {
        **base,
        "ok": True,
        "words": len(report.split()),
        "citations": len(citations),
        "citation_valid_rate": round(sum(1 for c in citations if c in ids) / len(citations), 4)
        if citations
        else 0.0,
        "cited_sentences": len(sentences),
        "citation_support_rate": round(supported / len(sentences), 4) if sentences else 0.0,
        "unverified_numbers": len(env.get("unverified_numbers") or []),
        "sources": len(sources),
        "domains": len({_host(s.get("url", "")) for s in sources} - {""}),
    }


def load_records(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(run_dir.glob("*.json")) if p.name[0] != "_"
    ]


NUMERIC = (
    "words",
    "citations",
    "citation_valid_rate",
    "citation_support_rate",
    "unverified_numbers",
    "sources",
    "domains",
    "cost_offpeak_usd",
    "wall_s",
)


def summarize(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean (and spread) of every numeric metric per depth and variant, plus the failure count."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for m in metrics:
        groups.setdefault(f"{m['variant']}/{m['depth']}", []).append(m)
    out: dict[str, Any] = {}
    for key, items in sorted(groups.items()):
        ok = [m for m in items if m.get("ok")]
        row: dict[str, Any] = {"runs": len(items), "failed": len(items) - len(ok)}
        for name in NUMERIC:
            values = [float(m[name]) for m in (ok if name not in ("cost_offpeak_usd", "wall_s") else items)]
            if values:
                row[name] = {
                    "mean": round(statistics.fmean(values), 4),
                    "stdev": round(statistics.pstdev(values), 4),
                }
        out[key] = row
    return out


def main() -> None:
    run_dir = Path(sys.argv[1])
    metrics = [record_metrics(r) for r in load_records(run_dir)]
    (run_dir / "_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    summary = summarize(metrics)
    (run_dir / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    json.dump(summary, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
