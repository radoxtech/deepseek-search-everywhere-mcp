"""Run a batch of research jobs, one fresh process per job, and save one record per job.

Examples (needs DEEPSEEK_API_KEY; costs money, so run only after the estimate is approved)::

    python evals/run.py --estimate --depths low,medium,high,max --repeats 2
    python evals/run.py --name baseline --depths low,medium,high,max --repeats 2
    python evals/run.py --name aa --depths medium --repeats 4
    python evals/run.py --name degraded --variant degraded-two-sources --depths medium --repeats 2

Records go to ``evals/results/<name>/<topic_id>__<variant>__<depth>__r<repeat>.json``; an existing
record is skipped, so an interrupted batch resumes. ``--max-usd`` stops starting new jobs once the
off-peak cost of the batch reaches it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import metrics  # noqa: E402

# Off-peak cost and time per depth measured for deepseek_research 3.x (README table).
DEPTH_COST_USD = {"low": 0.003, "medium": 0.008, "high": 0.012, "max": 0.031}
DEPTH_TIME_S = {"low": 6, "medium": 29, "high": 39, "max": 110}


def load_topics(selected: str, file: str = "topics.json") -> list[dict[str, str]]:
    topics = json.loads((HERE / file).read_text(encoding="utf-8"))["topics"]
    if selected == "all":
        return topics
    wanted = set(selected.split(","))
    return [t for t in topics if t["id"] in wanted]


def frozen_record(name: str, topic_id: str, depth: str, repeat: int) -> Path | None:
    """The saved record of run ``name`` whose sources a frozen job replays (None when there is none)."""
    matches = sorted((HERE / "results" / name).glob(f"{topic_id}__*__{depth}__r{repeat}.json"))
    return matches[0] if matches else None


def plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs = []
    for repeat in range(1, args.repeats + 1):
        for topic in load_topics(args.topics, args.topics_file):
            for depth in args.depths.split(","):
                job = {
                    "topic_id": topic["id"],
                    "topic": topic["topic"],
                    "depth": depth,
                    "variant": args.variant,
                    "repeat": repeat,
                }
                if getattr(args, "frozen_from", None):
                    frozen = frozen_record(args.frozen_from, topic["id"], depth, repeat)
                    if frozen is None:
                        continue
                    job["frozen_from"] = str(frozen)
                jobs.append(job)
    return jobs


def record_path(out: Path, job: dict[str, Any]) -> Path:
    return out / f"{job['topic_id']}__{job['variant']}__{job['depth']}__r{job['repeat']}.json"


def run_job(job: dict[str, Any], out: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(HERE / "run_one.py"),
        "--topic-id",
        job["topic_id"],
        "--topic",
        job["topic"],
        "--depth",
        job["depth"],
        "--variant",
        job["variant"],
        "--repeat",
        str(job["repeat"]),
    ]
    if job.get("frozen_from"):
        cmd += ["--frozen-from", job["frozen_from"]]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=False)
    if proc.returncode != 0:
        record = {
            **job,
            "envelope": {"ok": False, "error": {"reason": "harness_error"}},
            "stderr": proc.stderr[-4000:],
        }
    else:
        record = json.loads(proc.stdout)
    record_path(out, job).write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--topics", default="all")
    parser.add_argument("--topics-file", default="topics.json", help="topic list inside evals/")
    parser.add_argument("--depths", default="low,medium,high,max")
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--frozen-from",
        help="replay the sources of this results run (same topic, depth, repeat): no search or fetch cost",
    )
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--max-usd", type=float, default=5.0)
    parser.add_argument("--estimate", action="store_true", help="print the planned jobs, cost and time only")
    parser.add_argument(
        "--time-budget",
        type=float,
        default=None,
        help="seconds; start no job that may not finish in time (for shells that stop long commands)",
    )
    args = parser.parse_args()

    jobs = plan(args)
    if args.estimate:
        cost = sum(DEPTH_COST_USD[j["depth"]] for j in jobs)
        seconds = sum(DEPTH_TIME_S[j["depth"]] for j in jobs)
        print(
            json.dumps(
                {
                    "jobs": len(jobs),
                    "estimated_offpeak_usd": round(cost, 3),
                    "estimated_peak_usd_upper_bound": round(2 * cost, 3),
                    "estimated_minutes_sequential": round(seconds / 60, 1),
                    "estimated_minutes_parallel": round(seconds / 60 / max(1, args.parallel), 1),
                },
                indent=2,
            )
        )
        return

    out = HERE / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    pending = [j for j in jobs if not record_path(out, j).exists()]
    spent = sum(metrics.offpeak_cost(r) for r in metrics.load_records(out))
    print(f"{len(jobs)} jobs, {len(jobs) - len(pending)} already done, ${spent:.4f} spent", file=sys.stderr)
    started = time.monotonic()

    def fits(job: dict[str, Any]) -> bool:
        if args.time_budget is None:
            return True
        return time.monotonic() - started + 1.3 * DEPTH_TIME_S[job["depth"]] < args.time_budget

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = {}
        queue = list(pending)
        while queue or futures:
            while queue and len(futures) < args.parallel and spent < args.max_usd:
                fitting = next((j for j in queue if fits(j)), None)
                if fitting is None:
                    break
                queue.remove(fitting)
                job = fitting
                futures[pool.submit(run_job, job, out)] = job
            if not futures:
                reason = "time budget" if spent < args.max_usd else f"--max-usd {args.max_usd}"
                print(f"stopped: {len(queue)} job(s) left ({reason}), ${spent:.4f} spent", file=sys.stderr)
                break
            done = next(as_completed(futures))
            job = futures.pop(done)
            record = done.result()
            spent += metrics.offpeak_cost(record)
            m = metrics.record_metrics(record)
            print(
                f"{job['topic_id']} {job['depth']} r{job['repeat']}: ok={m['ok']} ${spent:.4f}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
