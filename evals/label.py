"""Label query-lab packs with a fixed, mechanical rule, in one pass per pack.

The label is a measurement, so the rule is stated once and applied identically: a key point
counts only when the page states all of its parts, including its figures. Batches of four pages
keep the prompt small, and a batch whose reply loses a page is retried, so a pack can never be
scored on a partial verdict.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

BATCH = 4
RULE = """You label web pages against key points of a research rubric.

A key point counts ONLY when the page itself states every part of it: the same fact AND, where the
key point carries figures, those figures at the same precision. A page that states part of a key
point, or a different figure, does not count. Judge only from the page text shown; never use outside
knowledge.

Return only this JSON object, with one key per page id shown, and no other text:
{"pages": {"<page id>": ["k1", "k4"], "<page id>": []}}
"""


def _parse_pack(path: Path) -> tuple[str, dict[str, tuple[str, str]], dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    topic = re.search(r"# Topic\n(.+?)\n", text)
    points = {k: (imp, pt) for k, imp, pt in re.findall(r"- (k\d+) \((\w+)\): (.+)", text)}
    pages: dict[str, str] = {}
    for match in re.finditer(
        r"^## Page ([0-9a-f]{10})\n(.*?)(?=^## Page [0-9a-f]{10}\n|\Z)", text, flags=re.MULTILINE | re.DOTALL
    ):
        pages[match.group(1)] = match.group(2).strip()
    return (topic.group(1) if topic else ""), points, pages


async def _label_batch(
    topic: str, points: dict[str, tuple[str, str]], batch: dict[str, str]
) -> dict[str, list[str]]:
    from deepseek_search_everywhere_mcp import deep_research as dr
    from deepseek_search_everywhere_mcp import runtime

    rubric = "\n".join(f"- {k} ({imp}): {pt}" for k, (imp, pt) in points.items())
    body = "\n\n".join(f"## Page {pid}\n{text}" for pid, text in batch.items())
    user = f"# Topic\n{topic}\n\n# Key points\n{rubric}\n\n# Pages\n{body}"
    for attempt in range(2):
        res = await runtime._chat_request(
            messages=[{"role": "system", "content": RULE}, {"role": "user", "content": user}],
            thinking={"type": "disabled"},
            model=runtime.REWRITE_MODEL,
            temperature=0.0,
            max_tokens=2048,
            response_format={"type": "json_object"},
            timeout=120.0,
            deadline=120.0,
            attempts=2,
        )
        parsed, _ = dr.sanitize_json(res.get("content") or "")
        pages = parsed.get("pages") if isinstance(parsed, dict) else None
        if isinstance(pages, dict) and all(pid in pages for pid in batch):
            return {pid: [str(k) for k in ids if str(k) in points] for pid, ids in pages.items()}
        if attempt == 0:
            user += "\n\nReturn one key for EVERY page id above, even when its list is empty."
    raise RuntimeError(f"batch of {len(batch)} pages came back incomplete twice")


async def _label_group(topic: str, points: dict, pages: dict[str, str]) -> dict[str, list[str]]:
    """Label a group, halving it when the model keeps dropping a page instead of failing the run."""
    try:
        return await _label_batch(topic, points, pages)
    except RuntimeError:
        ids = list(pages)
        if len(ids) == 1:
            raise
        mid = len(ids) // 2
        out = await _label_group(topic, points, {pid: pages[pid] for pid in ids[:mid]})
        out.update(await _label_group(topic, points, {pid: pages[pid] for pid in ids[mid:]}))
        return out


async def label_pack(pack: Path, out: Path) -> int:
    topic, points, pages = _parse_pack(pack)
    ids = list(pages)
    verdict: dict[str, list[str]] = {}
    for start in range(0, len(ids), BATCH):
        batch = {pid: pages[pid] for pid in ids[start : start + BATCH]}
        verdict.update(await _label_group(topic, points, batch))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"pages": verdict}, indent=1), "utf-8")
    return len(verdict)


async def main_async(args: argparse.Namespace) -> None:
    root = HERE / "results" / args.name
    for pack in sorted(root.glob("*/packs/*.md")):
        out = pack.parent.parent / "verdicts" / f"{pack.stem}.json"
        if out.exists() and not args.force:
            print(pack.parent.parent.name, pack.stem, "already labelled", flush=True)
            continue
        n = await label_pack(pack, out)
        print(pack.parent.parent.name, pack.stem, n, "pages", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--force", action="store_true")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
