#!/usr/bin/env python3
"""
Turn a grader's downloaded votes file into preference records.

The HTML page knows only "A" and "B". This resolves those back to models via
the bundle's mapping file, which stayed on this machine, and writes records in
the same shape vote_pairwise.py and chat_vote.py produce so router.py reads
all three identically.

Usage:
    python ingest_votes.py votes_advisory_nikita.json
    python ingest_votes.py votes_*.json          # several graders at once
"""

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STATE_DIR = Path(__file__).parent / ".vote_state"


def load_state(bundle_id):
    path = STATE_DIR / f"bundle_{bundle_id}.json"
    if not path.exists():
        sys.exit(f"no mapping for bundle {bundle_id} at {path}.\n"
                 "That file is what holds the A/B -> model identities; without "
                 "it the votes cannot be resolved and are not recoverable from "
                 "the votes file alone.")
    return json.loads(path.read_text(encoding="utf-8"))


def ingest_one(path, preference_dir, dry_run=False):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("bundle_id", "votes", "grader"):
        if key not in payload:
            sys.exit(f"{path}: not a votes file (missing {key!r})")

    state = load_state(payload["bundle_id"])
    category = state.get("category", payload.get("category", "advisory"))
    mapping = state["mapping"]

    out_dir = Path(preference_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for task_id, votes in sorted(payload["votes"].items()):
        if task_id not in mapping:
            print(f"  skip {task_id}: not in this bundle", file=sys.stderr)
            continue
        sides = mapping[task_id]
        record = {
            "task_id": task_id,
            "date": payload.get("submitted") or datetime.now(timezone.utc).isoformat(),
            "transcript_a_file": sides["a"]["file"],
            "transcript_b_file": sides["b"]["file"],
            "model_a": f"{sides['a']['provider']}/{sides['a']['model']}",
            "model_b": f"{sides['b']['provider']}/{sides['b']['model']}",
            "votes": votes,
            "elicited_via": "html",
            "grader": payload["grader"],
            "bundle_id": payload["bundle_id"],
        }
        if payload.get("notes", {}).get(task_id):
            record["note"] = payload["notes"][task_id]

        # Grader name is part of the filename so several people voting on the
        # same bundle produce separate records instead of overwriting each
        # other -- inter-rater disagreement is data, not a collision.
        safe_grader = "".join(c if c.isalnum() else "_" for c in payload["grader"])
        prefix = "personal_" if category == "personal" else ""
        name = f"{prefix}{task_id}__{safe_grader}__{payload['bundle_id']}.json"
        dest = out_dir / name

        if dest.exists():
            print(f"  skip {task_id}: {dest.name} already ingested", file=sys.stderr)
            continue
        if not dry_run:
            dest.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        written.append((task_id, record["model_a"], record["model_b"], dest.name))

    print(f"{Path(path).name}: grader {payload['grader']!r}, "
          f"{len(written)} record(s){' (dry run)' if dry_run else ''}")
    for task_id, ma, mb, name in written:
        print(f"    {task_id}: A={ma}  B={mb}  -> {name}")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--preference-dir", default="preference")
    ap.add_argument("--dry-run", action="store_true",
                     help="resolve and print without writing, to check a file "
                          "before it becomes part of the record")
    args = ap.parse_args()

    expanded = []
    for p in args.paths:
        hits = glob.glob(p)
        if not hits:
            sys.exit(f"no such file: {p}")
        expanded.extend(hits)

    total = 0
    for p in sorted(expanded):
        total += len(ingest_one(p, args.preference_dir, args.dry_run))
    print(f"\n{total} preference record(s) "
          f"{'would be written' if args.dry_run else 'written'}.")
    if not args.dry_run and total:
        print("Now: python router.py rank --category advisory")


if __name__ == "__main__":
    main()
