#!/usr/bin/env python3
"""
Blind pairwise voting driven from a chat session instead of a terminal prompt.

Same protocol and same output format as vote_pairwise.py: side assignment is
randomised, the A/B -> model mapping is written to a state file and never
printed, and the vote record lands in /preference identically.

    python chat_vote.py show   --task-id advisory_001
    python chat_vote.py record --task-id advisory_001 --votes a,b,tie,a,a
"""

import argparse
import glob
import json
import pickle
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STATE_DIR = Path(".vote_state")
DEFAULT_DIMENSIONS = ["held_position", "specificity", "context_retention", "honesty", "overall"]


def load_dimensions(task_path):
    if not task_path or not Path(task_path).exists():
        return DEFAULT_DIMENSIONS
    with open(task_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    dims = data.get("dimensions") if isinstance(data, dict) else None
    if not dims:
        return DEFAULT_DIMENSIONS
    names = []
    for d in dims:
        if isinstance(d, dict):
            names.extend(d.keys())
        elif isinstance(d, str):
            names.append(d)
    return names or DEFAULT_DIMENSIONS


def find_pair(task_id):
    files = sorted(glob.glob(f"runs/{task_id}__*.json"))
    if len(files) != 2:
        sys.exit(f"expected exactly 2 transcripts for {task_id}, found {len(files)}: {files}")
    return [Path(p) for p in files]


def cmd_show(args):
    files = find_pair(args.task_id)
    transcripts = [json.load(open(f, encoding="utf-8")) for f in files]
    if transcripts[0]["task_id"] != transcripts[1]["task_id"]:
        sys.exit("transcripts are for different tasks")

    order = [0, 1]
    if random.SystemRandom().random() < 0.5:
        order = [1, 0]

    STATE_DIR.mkdir(exist_ok=True)
    with open(STATE_DIR / f"{args.task_id}.pkl", "wb") as f:
        pickle.dump({"files": [str(p) for p in files], "order": order}, f)

    print(f"Task: {args.task_id}   (identities hidden)")
    for label, idx in zip("AB", order):
        t = transcripts[idx]
        print(f"\n{'=' * 30} TRANSCRIPT {label} {'=' * 30}")
        for i, ex in enumerate(t["exchanges"], 1):
            print(f"\n----- turn {i} -----")
            print(f"[user]\n{ex['user'].strip()}")
            print(f"\n[model {label}]\n{ex['assistant'].strip()}")

    dims = load_dimensions(args.task_file)
    print(f"\nDimensions to rank: {', '.join(dims)}")


def cmd_record(args):
    state_file = STATE_DIR / f"{args.task_id}.pkl"
    if not state_file.exists():
        sys.exit(f"no shown-state for {args.task_id}; run `show` first")
    state = pickle.load(open(state_file, "rb"))

    files = [Path(p) for p in state["files"]]
    transcripts = [json.load(open(f, encoding="utf-8")) for f in files]
    order = state["order"]
    shown = [transcripts[order[0]], transcripts[order[1]]]
    shown_files = [files[order[0]], files[order[1]]]

    dims = load_dimensions(args.task_file)
    raw = [v.strip().lower() for v in args.votes.split(",")]
    if len(raw) != len(dims):
        sys.exit(f"expected {len(dims)} votes for {dims}, got {len(raw)}")
    for v in raw:
        if v not in ("a", "b", "tie"):
            sys.exit(f"invalid vote {v!r}; use a, b, or tie")
    votes = dict(zip(dims, raw))

    identity_a = f"{shown[0]['provider']}/{shown[0]['model']}"
    identity_b = f"{shown[1]['provider']}/{shown[1]['model']}"

    record = {
        "task_id": args.task_id,
        "date": datetime.now(timezone.utc).isoformat(),
        "transcript_a_file": str(shown_files[0]),
        "transcript_b_file": str(shown_files[1]),
        "model_a": identity_a,
        "model_b": identity_b,
        "votes": votes,
        "elicited_via": "chat",
    }

    prefix = "personal_" if args.category == "personal" else ""
    stamp = record["date"].replace(":", "-")
    out_dir = Path(args.preference_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{prefix}{args.task_id}__{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    state_file.unlink()
    print(f"Revealed: A = {identity_a}, B = {identity_b}")
    print(json.dumps(votes, indent=2))
    print(f"Vote written to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("show")
    s.add_argument("--task-id", required=True)
    s.add_argument("--task-file")
    s.set_defaults(func=cmd_show)

    r = sub.add_parser("record")
    r.add_argument("--task-id", required=True)
    r.add_argument("--task-file")
    r.add_argument("--votes", required=True,
                   help="comma-separated a/b/tie, one per dimension, in dimension order")
    r.add_argument("--category", default="advisory",
                   choices=["advisory", "research", "documents", "personal"])
    r.add_argument("--preference-dir", default="preference")
    r.set_defaults(func=cmd_record)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
