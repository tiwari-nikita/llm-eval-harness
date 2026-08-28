#!/usr/bin/env python3
"""
Build a self-contained HTML grading page from run transcripts.

The blind pairwise vote is the one judgment in this harness a model cannot
make. It was also the step that stalled for days, because casting a vote meant
opening a terminal. This produces a single .html file that opens from
file://, needs no server, no install and no account, and downloads a small
JSON when the grader is done. Send one file, get one file back.

Blinding is enforced by construction rather than by discipline: the bundle
embedded in the page contains no model names, only side "A" and "B". The
A/B -> model mapping is written to a separate local file that the grader
never receives, and ingest_votes.py is what puts the two back together.

Privacy: the personal category is real private chat history. It is excluded
unless --include-personal is passed explicitly, and even then the generated
page is gitignored like everything else derived from /runs.

Usage:
    python make_grading_bundle.py --category advisory
    python make_grading_bundle.py --category advisory -o grade_advisory.html
"""

import argparse
import glob
import json
import random
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STATE_DIR = Path(__file__).parent / ".vote_state"
TEMPLATE = Path(__file__).parent / "grade_template.html"
DEFAULT_DIMENSIONS = ["held_position", "specificity", "context_retention",
                      "honesty", "overall"]


def load_dimensions(task_file):
    """Dimension names plus their one-line descriptions, when the task has them."""
    if not task_file or not Path(task_file).exists():
        return [{"name": d, "hint": ""} for d in DEFAULT_DIMENSIONS]
    with open(task_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    raw = data.get("dimensions") if isinstance(data, dict) else None
    if not raw:
        return [{"name": d, "hint": ""} for d in DEFAULT_DIMENSIONS]
    out = []
    for item in raw:
        if isinstance(item, dict):
            for k, v in item.items():
                out.append({"name": k, "hint": v if isinstance(v, str) else ""})
        else:
            out.append({"name": str(item), "hint": ""})
    return out or [{"name": d, "hint": ""} for d in DEFAULT_DIMENSIONS]


def find_pairs(runs_dir, category):
    """Group transcripts by task id, keeping only tasks with exactly two."""
    by_task = {}
    for path in sorted(glob.glob(str(Path(runs_dir) / f"{category}_*.json"))):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not data.get("exchanges"):
            print(f"  skip {Path(path).name}: error record, nothing to grade",
                  file=sys.stderr)
            continue
        by_task.setdefault(data["task_id"], []).append((path, data))

    pairs = []
    for task_id, entries in sorted(by_task.items()):
        if len(entries) != 2:
            print(f"  skip {task_id}: found {len(entries)} healthy transcripts, "
                  f"need exactly 2", file=sys.stderr)
            continue
        pairs.append((task_id, entries))
    return pairs


def build(args):
    pairs = find_pairs(args.runs_dir, args.category)
    if not pairs:
        sys.exit(f"no gradeable {args.category} pairs in {args.runs_dir}")

    dimensions = load_dimensions(args.task_file)
    bundle_id = uuid.uuid4().hex[:12]
    rng = random.SystemRandom()

    tasks, mapping = [], {}
    for task_id, entries in pairs:
        order = [0, 1]
        if rng.random() < 0.5:
            order = [1, 0]
        shown = [entries[order[0]], entries[order[1]]]

        tasks.append({
            "task_id": task_id,
            "turns": [
                {"user": e["user"], "a": shown[0][1]["exchanges"][i]["assistant"],
                 "b": shown[1][1]["exchanges"][i]["assistant"]}
                for i, e in enumerate(shown[0][1]["exchanges"])
            ],
            # surfaced so a grader can see a reply was cut off by the cap
            # rather than mistaking it for the model trailing away
            "a_truncated": [i + 1 for i, e in enumerate(shown[0][1]["exchanges"])
                            if e.get("truncated")],
            "b_truncated": [i + 1 for i, e in enumerate(shown[1][1]["exchanges"])
                            if e.get("truncated")],
        })
        mapping[task_id] = {
            "a": {"file": shown[0][0], "provider": shown[0][1]["provider"],
                  "model": shown[0][1]["model"]},
            "b": {"file": shown[1][0], "provider": shown[1][1]["provider"],
                  "model": shown[1][1]["model"]},
        }

    bundle = {
        "bundle_id": bundle_id,
        "category": args.category,
        "created": datetime.now(timezone.utc).isoformat(),
        "dimensions": dimensions,
        "tasks": tasks,
    }

    STATE_DIR.mkdir(exist_ok=True)
    state_path = STATE_DIR / f"bundle_{bundle_id}.json"
    state_path.write_text(json.dumps(
        {"bundle_id": bundle_id, "category": args.category, "mapping": mapping},
        ensure_ascii=False, indent=2), encoding="utf-8")

    if not TEMPLATE.exists():
        sys.exit(f"missing {TEMPLATE}")
    html = TEMPLATE.read_text(encoding="utf-8")
    # json.dumps escapes nothing that matters inside a <script> block except a
    # literal </script> appearing in transcript text, which would close the tag
    # early and break the page.
    payload = json.dumps(bundle, ensure_ascii=False).replace("</", "<\\/")
    html = html.replace("/*__BUNDLE__*/null", payload)

    out = Path(args.out or f"grade_{args.category}.html")
    out.write_text(html, encoding="utf-8")

    print(f"wrote {out}  ({len(tasks)} pairs, bundle {bundle_id})")
    print(f"identities kept in {state_path} — do not send that file")
    print(f"\nSend {out} to a grader. When they send back the JSON:")
    print(f"    python ingest_votes.py <their-file.json>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="advisory",
                     choices=["advisory", "personal", "research", "documents"])
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--task-file", help="task yaml to read dimension names from; "
                                        "defaults to tasks/<category>.yaml")
    ap.add_argument("-o", "--out")
    ap.add_argument("--include-personal", action="store_true",
                     help="required to build a personal-category page, since "
                          "those transcripts are real private chat history")
    args = ap.parse_args()

    if args.category == "personal" and not args.include_personal:
        sys.exit("refusing to build a personal-category page without "
                 "--include-personal: those transcripts are real private chat "
                 "history, and this file is meant to be sent to other people")
    if not args.task_file:
        guess = Path("tasks") / f"{args.category}.yaml"
        args.task_file = str(guess) if guess.exists() else None
    build(args)


if __name__ == "__main__":
    main()
