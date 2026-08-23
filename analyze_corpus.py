#!/usr/bin/env python3
"""
Bucket a public conversation corpus, and optionally compare it against your
own extracted prompts.

Reads WildChat / LMSYS-Chat-1M style records. Both store a `conversation`
list of {role, content} turns; this script is tolerant of the variations.

Install first (not available in every environment):
    pip install datasets pandas pyarrow

Examples:
    # from Hugging Face, streaming so you don't pull all 1M rows
    python analyze_corpus.py --hf allenai/WildChat-1M --limit 20000

    # from a local file you already downloaded
    python analyze_corpus.py --file wildchat.jsonl --limit 20000

    # compare with your own export
    python analyze_corpus.py --hf allenai/WildChat-1M --limit 20000 \
        --compare my_prompts.jsonl

Outputs a distribution table, per-bucket samples for hand-checking, and a
turn-count profile.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from taxonomy import classify, word_count, length_band

# Windows consoles default to cp1252, which chokes on curly quotes and
# other characters that show up constantly in real chat exports.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def user_turns(record):
    """Return the list of human-side turn texts from one conversation record."""
    convo = None
    for key in ("conversation", "conversations", "messages", "turns"):
        if isinstance(record.get(key), list):
            convo = record[key]
            break
    if convo is None:
        return []

    out = []
    for turn in convo:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or turn.get("from") or "").lower()
        if role not in ("user", "human"):
            continue
        content = turn.get("content") or turn.get("value") or turn.get("text")
        if isinstance(content, list):
            content = " ".join(
                c if isinstance(c, str) else str(c.get("text", ""))
                for c in content
            )
        if isinstance(content, str) and content.strip():
            out.append(content.strip())
    return out


def iter_hf(name, limit, split):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets")
    ds = load_dataset(name, split=split, streaming=True)
    for i, rec in enumerate(ds):
        if i >= limit:
            break
        yield rec


def iter_file(path, limit):
    p = Path(path)
    if p.suffix == ".jsonl":
        with p.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                yield json.loads(line)
    else:
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        for rec in data[:limit]:
            yield rec


def profile(records, sample_n=4):
    buckets = Counter()
    lengths = Counter()
    turn_counts = Counter()
    samples = defaultdict(list)
    total_turns = 0
    convos = 0

    for rec in records:
        turns = user_turns(rec)
        if not turns:
            continue
        convos += 1
        total_turns += len(turns)
        turn_counts[min(len(turns), 10)] += 1

        # classify the opening turn: it sets what the conversation is for
        first = turns[0]
        b = classify(first)
        buckets[b] += 1
        lengths[length_band(word_count(first))] += 1

        if len(samples[b]) < sample_n and random.random() < 0.3:
            samples[b].append(first[:220].replace("\n", " "))

    return {
        "convos": convos,
        "buckets": buckets,
        "lengths": lengths,
        "turn_counts": turn_counts,
        "avg_turns": total_turns / convos if convos else 0,
        "samples": samples,
    }


def table(counter, total, label):
    print(f"\n{label}")
    for k, n in counter.most_common():
        bar = "#" * int(40 * n / total) if total else ""
        print(f"  {str(k):<12} {n:>7}  {n/total:>6.1%}  {bar}")


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf", help="Hugging Face dataset id")
    src.add_argument("--file", help="local .json or .jsonl")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--compare", help="my_prompts.jsonl from extract_prompts.py")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    records = iter_hf(args.hf, args.limit, args.split) if args.hf \
        else iter_file(args.file, args.limit)

    p = profile(records)
    if not p["convos"]:
        sys.exit("No conversations parsed. Check the record shape against user_turns().")

    print(f"conversations parsed: {p['convos']}")
    print(f"mean user turns per conversation: {p['avg_turns']:.2f}")

    table(p["buckets"], p["convos"], "opening-turn bucket")
    table(p["lengths"], p["convos"], "opening-turn length")
    table(p["turn_counts"], p["convos"], "user turns per conversation (10 = 10+)")

    print("\nsamples per bucket, hand-check these before trusting the table")
    for b, ss in p["samples"].items():
        print(f"\n  [{b}]")
        for s in ss:
            print(f"    - {s}")

    if args.compare:
        mine = Counter()
        n = 0
        with open(args.compare, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                mine[classify(rec["text"])] += 1
                n += 1
        if n:
            print("\n\nyou vs population (percentage points, + means you do more)")
            keys = set(mine) | set(p["buckets"])
            rows = []
            for k in keys:
                mine_pct = mine[k] / n
                pop_pct = p["buckets"][k] / p["convos"]
                rows.append((mine_pct - pop_pct, k, mine_pct, pop_pct))
            for diff, k, m, pp in sorted(rows, reverse=True):
                print(f"  {k:<12} you {m:>6.1%}   pop {pp:>6.1%}   {diff:+.1%}")
            print("\nThe biggest positive gaps are where a generic router would "
                  "misserve you. Weight those categories up.")


if __name__ == "__main__":
    main()
