#!/usr/bin/env python3
"""
Blind pairwise vote between two transcripts of the same task.

Model identities are hidden as "A" and "B" until after the vote is written.
Side assignment (which transcript is shown as A) is randomised per vote, so
reading order can't correlate with a model.

Usage:
    python vote_pairwise.py --transcripts runs/advisory_001__groq_llama.json runs/advisory_001__openrouter_gemma.json
"""

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_DIMENSIONS = ["held_position", "specificity", "context_retention", "honesty", "overall"]


def load_dimensions(task_path):
    if not task_path:
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


def print_transcript(label, transcript):
    print(f"\n{'=' * 10} Transcript {label} {'=' * 10}")
    for i, ex in enumerate(transcript["exchanges"], 1):
        print(f"\n--- turn {i} ---")
        print(f"[user]: {ex['user'].strip()}")
        print(f"[model]: {ex['assistant'].strip()}")


def ask_dimension(dim, label_a, label_b):
    while True:
        ans = input(f"  {dim} — A, B, or tie? ").strip().lower()
        if ans in ("a", "b", "tie"):
            return ans
        print("  enter 'a', 'b', or 'tie'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", nargs=2, required=True,
                     metavar=("TRANSCRIPT_A_FILE", "TRANSCRIPT_B_FILE"))
    ap.add_argument("--task-file", help="task yaml to pull dimension names from "
                     "(defaults to the standard advisory dimensions)")
    ap.add_argument("--preference-dir", default="preference")
    ap.add_argument("--category", default="advisory",
                     choices=["advisory", "research", "documents", "personal"])
    ap.add_argument("--seed", type=int, default=None,
                     help="fix the A/B randomisation, mainly for testing")
    args = ap.parse_args()

    files = [Path(p) for p in args.transcripts]
    transcripts = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            transcripts.append(json.load(fh))

    if transcripts[0]["task_id"] != transcripts[1]["task_id"]:
        sys.exit(f"transcripts are for different tasks: "
                 f"{transcripts[0]['task_id']} vs {transcripts[1]['task_id']}")

    rng = random.Random(args.seed)
    order = [0, 1]
    if rng.random() < 0.5:
        order = [1, 0]
    shown = [transcripts[order[0]], transcripts[order[1]]]
    shown_files = [files[order[0]], files[order[1]]]

    print(f"Task: {transcripts[0]['task_id']}  (identities hidden until vote is recorded)")
    print_transcript("A", shown[0])
    print_transcript("B", shown[1])

    dimensions = load_dimensions(args.task_file)
    print(f"\nRank on each dimension: {', '.join(dimensions)}")
    votes = {dim: ask_dimension(dim, "A", "B") for dim in dimensions}

    identity_a = f"{shown[0]['provider']}/{shown[0]['model']}"
    identity_b = f"{shown[1]['provider']}/{shown[1]['model']}"
    print(f"\nRevealed: A = {identity_a}, B = {identity_b}")

    record = {
        "task_id": transcripts[0]["task_id"],
        "date": datetime.now(timezone.utc).isoformat(),
        "transcript_a_file": str(shown_files[0]),
        "transcript_b_file": str(shown_files[1]),
        "model_a": identity_a,
        "model_b": identity_b,
        "votes": votes,
    }

    prefix = "personal_" if args.category == "personal" else ""
    stamp = record["date"].replace(":", "-")
    out_name = f"{prefix}{record['task_id']}__{stamp}.json"
    out_dir = Path(args.preference_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / out_name, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"\nVote written to {out_dir / out_name}")


if __name__ == "__main__":
    main()
