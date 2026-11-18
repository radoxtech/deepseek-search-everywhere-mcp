"""Run one research job in this process and print its record as JSON on stdout.

``run.py`` starts this script once per job, so every job runs in a fresh process with empty caches and
job store. The record holds the full tool envelope, the debug dump (``DEEPSEEK_MCP_DEBUG_DUMP_DIR``),
the token usage of every chat-completions call and the wall time. Needs ``DEEPSEEK_API_KEY``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import variants  # noqa: E402  (harness-local module next to this script)


async def _run(args: argparse.Namespace, dump_dir: Path) -> dict[str, Any]:
    os.environ["DEEPSEEK_MCP_DEBUG_DUMP_DIR"] = str(dump_dir)
    from deepseek_search_everywhere_mcp import runtime, server

    usage: list[dict[str, Any]] = []
    chat = runtime._chat_request

    async def recording_chat(*a: Any, **kw: Any) -> dict[str, Any]:
        res = await chat(*a, **kw)
        usage.append({"ok": bool(res.get("ok")), "usage": res.get("usage") or {}, "model": kw.get("model")})
        return res

    runtime._chat_request = recording_chat
    if args.frozen_from:
        variants.freeze(args.frozen_from)
    variants.apply(args.variant)

    started = time.monotonic()
    tool = server.deepseek_research
    res = await tool(args.topic, depth=args.depth, refresh=True)
    polls = 0
    while res.get("ok") and res.get("status") == "running" and polls < 60:
        polls += 1
        res = await server.deepseek_check_research(res["job_id"])
    wall_s = time.monotonic() - started

    dump: dict[str, Any] | None = None
    job_id = res.get("job_id")
    if job_id and (dump_dir / f"{job_id}.json").exists():
        dump = json.loads((dump_dir / f"{job_id}.json").read_text(encoding="utf-8"))
    elif len(list(dump_dir.glob("*.json"))) == 1:  # inline results carry no job_id
        dump = json.loads(next(dump_dir.glob("*.json")).read_text(encoding="utf-8"))
    return {
        "topic_id": args.topic_id,
        "topic": args.topic,
        "tool": args.tool,
        "depth": args.depth,
        "variant": args.variant,
        "repeat": args.repeat,
        "frozen_from": args.frozen_from,
        "wall_s": round(wall_s, 3),
        "envelope": res,
        "dump": dump,
        "chat_usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic-id", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--depth", required=True, choices=["low", "medium", "high", "max"])
    parser.add_argument("--variant", default="baseline", choices=sorted(variants.VARIANTS))
    parser.add_argument("--tool", default="research", choices=["research"])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--frozen-from", help="saved record whose sources to replay (no search, no fetch)")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="evals-dump-") as tmp:
        record = asyncio.run(_run(args, Path(tmp)))
    json.dump(record, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
