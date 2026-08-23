#!/usr/bin/env python3
"""
Pull your own messages out of a chat export.

Works on Claude / ChatGPT / Gemini exports without knowing their exact schema:
it walks the JSON recursively and collects any object that looks like a message
with a human-side role.

Usage:
    python extract_prompts.py conversations.json -o my_prompts.jsonl
    python extract_prompts.py conversations.json --min-words 8 --multiturn-only

Nothing leaves your machine. Do not commit the output.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from taxonomy import classify, word_count

HUMAN_ROLES = {"human", "user", "you", "prompt"}
ROLE_KEYS = ("role", "sender", "author", "from", "speaker")
TEXT_KEYS = ("text", "content", "message", "body", "value", "prompt")


def role_of(obj):
    """Return a lowercased role string if this dict declares one."""
    for k in ROLE_KEYS:
        v = obj.get(k)
        if isinstance(v, str):
            return v.strip().lower()
        # ChatGPT nests: {"author": {"role": "user"}}
        if isinstance(v, dict):
            inner = v.get("role")
            if isinstance(inner, str):
                return inner.strip().lower()
    return None


def text_of(obj):
    """Pull the message text out, handling the common nested shapes."""
    for k in TEXT_KEYS:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v
        # content can be a list of blocks
        if isinstance(v, list):
            parts = []
            for item in v:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    for tk in ("text", "value", "content"):
                        if isinstance(item.get(tk), str):
                            parts.append(item[tk])
                            break
            joined = "\n".join(p for p in parts if p.strip())
            if joined.strip():
                return joined
        # ChatGPT: {"content": {"parts": [...]}}
        if isinstance(v, dict) and isinstance(v.get("parts"), list):
            parts = [p for p in v["parts"] if isinstance(p, str)]
            joined = "\n".join(parts)
            if joined.strip():
                return joined
    return None


def walk(node, out, convo_title=None, depth=0):
    """Recursively collect human messages, tagging each with its conversation."""
    if depth > 40:
        return

    if isinstance(node, dict):
        title = node.get("name") or node.get("title") or convo_title

        r = role_of(node)
        if r in HUMAN_ROLES:
            t = text_of(node)
            if t:
                out.append({"conversation": title, "text": t.strip()})

        for v in node.values():
            if isinstance(v, (dict, list)):
                walk(v, out, title, depth + 1)

    elif isinstance(node, list):
        for item in node:
            walk(item, out, convo_title, depth + 1)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("my_prompts.jsonl"))
    ap.add_argument("--min-words", type=int, default=5,
                    help="drop very short turns like 'yes' or 'go on'")
    ap.add_argument("--multiturn-only", action="store_true",
                    help="keep only conversations with 4+ of your turns")
    args = ap.parse_args()

    if not args.export.exists():
        sys.exit(f"no such file: {args.export}")

    with args.export.open(encoding="utf-8") as f:
        data = json.load(f)

    msgs = []
    walk(data, msgs)

    if not msgs:
        sys.exit(
            "Found no human-side messages. The export schema may differ from "
            "what this script expects. Open the JSON, find how a user message "
            "is labelled, and add that key to HUMAN_ROLES or ROLE_KEYS."
        )

    # dedupe while preserving order
    seen = set()
    unique = []
    for m in msgs:
        key = (m["conversation"], m["text"])
        if key not in seen:
            seen.add(key)
            unique.append(m)

    kept = [m for m in unique if word_count(m["text"]) >= args.min_words]

    if args.multiturn_only:
        counts = Counter(m["conversation"] for m in kept)
        kept = [m for m in kept if counts[m["conversation"]] >= 4]

    for m in kept:
        m["words"] = word_count(m["text"])
        m["bucket"] = classify(m["text"])

    with args.out.open("w", encoding="utf-8") as f:
        for m in kept:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    buckets = Counter(m["bucket"] for m in kept)
    convos = len({m["conversation"] for m in kept})

    print(f"messages found:      {len(unique)}")
    print(f"kept after filters:  {len(kept)}")
    print(f"conversations:       {convos}")
    print("\nby bucket:")
    for b, n in buckets.most_common():
        print(f"  {b:<10} {n:>5}  ({n / len(kept):.0%})")
    print(f"\nwritten to {args.out}")
    print("Add this file to .gitignore.")


if __name__ == "__main__":
    main()
