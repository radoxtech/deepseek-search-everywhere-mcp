"""Whole-context test: does the writer do better with every found page in full than with the top 10?

The model has a 1M-token window and the medium context uses about 40k tokens of it. This harness takes
the pages a query-lab run found (``evals/query_lab.py run``), fetches every one in full, and replays two
frozen synthesis contexts through real research jobs (``run_one.py --frozen-from``)::

    python evals/full_context.py fetch --name qlab              # full pages; about 24 fetches per topic
    python evals/full_context.py records --name qlab            # the two frozen contexts per topic
    python evals/full_context.py run --name qlab --repeats 2    # jobs into results/fullctx-{top10,all}

Then judge both result sets with ``evals/keypoints.py build --side fullctx-top10:medium --side
fullctx-all:medium --out kp-fullctx``. ``top10`` is today's medium context (top 10 ranked pages,
per-page and total caps); ``all`` is every page, each up to the per-page cap, with no total cap.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import query_lab  # noqa: E402

from deepseek_search_everywhere_mcp import deep_research as dr  # noqa: E402

CONTEXTS = ("top10", "all")


def _topic_dirs(name: str, style: str) -> list[Path]:
    root = HERE / "results" / name
    return [p for p in sorted(root.iterdir()) if p.is_dir() and (p / f"{style}.json").exists()]


def _lists(data: dict[str, Any]) -> list[list[dict[str, Any]]]:
    by_query: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for r in data["results"]:
        by_query.setdefault(r["query"], []).append(dict(r))
    return list(by_query.values())


async def _fetch_topic(data: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    from deepseek_search_everywhere_mcp.gap_fill import _fill_gaps

    cfg = dr.DEPTH_CONFIG["medium"]
    pool, _ = dr.rank_and_diversify_sources(_lists(data), len(data["results"]))
    for source in pool:
        source["search_content"] = source.get("content") or ""
        source["content"] = ""
    fill = await _fill_gaps(
        pool,
        cfg.per_page_char_cap,
        cfg.per_page_char_cap * len(pool),
        refresh=False,
        deadline=120.0,
        max_pages=len(pool),
        share_total=False,
    )
    for source in pool:  # a failed fetch keeps the search text
        if not source.get("content"):
            source["content"] = source.pop("search_content")
        else:
            source.pop("search_content", None)
    return pool, fill.requests_used


def fetch(args: argparse.Namespace) -> None:
    for topic_dir in _topic_dirs(args.name, args.style):
        if (topic_dir / "full_pages.json").exists():
            continue
        data = json.loads((topic_dir / f"{args.style}.json").read_text())
        pool, requests = asyncio.run(_fetch_topic(data))
        (topic_dir / "full_pages.json").write_text(json.dumps(pool, indent=1, ensure_ascii=False), "utf-8")
        chars = sum(len(s.get("content") or "") for s in pool)
        print(topic_dir.name, len(pool), "pages", chars, "chars", requests, "requests", flush=True)


def build_contexts(pool: list[dict[str, Any]], topic: str) -> dict[str, list[dict[str, Any]]]:
    cfg = dr.DEPTH_CONFIG["medium"]
    top = dr.select_synthesis_context([dict(s) for s in pool[: cfg.k_sources]], "medium", topic=topic)
    every = []
    for n, s in enumerate(pool, 1):
        entry = dict(s)
        entry["id"] = n
        entry["content"] = (entry.get("content") or "")[: cfg.per_page_char_cap]
        if entry["content"]:
            every.append(entry)
    return {"top10": top, "all": every}


def records(args: argparse.Namespace) -> None:
    for topic_dir in _topic_dirs(args.name, args.style):
        pool = json.loads((topic_dir / "full_pages.json").read_text())
        topic = query_lab._rubric(topic_dir.name)["topic"]
        for label, context in build_contexts(pool, topic).items():
            record = {"dump": {"synthesis_context": context}}
            (topic_dir / f"frozen_{label}.json").write_text(json.dumps(record, ensure_ascii=False), "utf-8")
            chars = sum(len(s["content"]) for s in context)
            print(topic_dir.name, label, len(context), "sources", chars, "chars")


def run(args: argparse.Namespace) -> None:
    """Start up to ``--parallel`` pending jobs at once and wait for them (one batch per call)."""
    pending = []
    for topic_dir in _topic_dirs(args.name, args.style):
        topic = query_lab._rubric(topic_dir.name)["topic"]
        for repeat in range(1, args.repeats + 1):
            for label in args.contexts.split(","):
                suffix = "" if args.variant == "baseline" else f"-{args.variant}"
                out = HERE / "results" / f"fullctx-{label}{suffix}"
                out.mkdir(exist_ok=True)
                path = out / f"{topic_dir.name}__baseline__medium__r{repeat}.json"
                if not path.exists():
                    pending.append((topic_dir, topic, label, repeat, path))
    batch = pending[: args.parallel]
    procs = []
    for topic_dir, topic, label, repeat, path in batch:
        cmd = [sys.executable, str(HERE / "run_one.py"), "--topic-id", topic_dir.name, "--topic", topic]
        cmd += ["--depth", "medium", "--repeat", str(repeat), "--variant", args.variant]
        cmd += ["--frozen-from", str(topic_dir / f"frozen_{label}.json")]
        procs.append(
            (
                topic_dir,
                label,
                repeat,
                path,
                subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True),
            )
        )
    for topic_dir, label, repeat, path, proc in procs:
        stdout, stderr = proc.communicate()
        if proc.returncode != 0 or not stdout.strip():
            print(topic_dir.name, label, repeat, "FAILED", stderr[-500:], flush=True)
            continue
        path.write_text(stdout, "utf-8")
        rec = json.loads(stdout)
        tokens = sum((u.get("usage") or {}).get("prompt_tokens", 0) for u in rec["chat_usage"])
        env = rec.get("envelope") or {}
        print(
            topic_dir.name,
            label,
            repeat,
            f"{rec['wall_s']:.0f}s",
            tokens,
            "prompt tokens",
            env.get("status"),
            flush=True,
        )
    print(len(pending) - len(batch), "jobs left")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, func in (("fetch", fetch), ("records", records), ("run", run)):
        p = sub.add_parser(name)
        p.add_argument("--name", required=True)
        p.add_argument("--style", default="aspects")
        p.add_argument("--repeats", type=int, default=2)
        p.add_argument("--parallel", type=int, default=4)
        p.add_argument("--variant", default="baseline")
        p.add_argument("--contexts", default=",".join(CONTEXTS))
        p.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
