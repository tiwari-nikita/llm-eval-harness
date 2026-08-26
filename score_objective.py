#!/usr/bin/env python3
"""
Grade objective-category transcripts (research/documents) against their
task's binary criteria and hard_fail list.

Two modes:
  grade        model-graded first pass, writes /scores/<task>__<model>.json
  hand-sample  pull a random 15% of already-graded transcripts for you to
               re-grade by hand, then report the disagreement rate

Score = met / total criteria, unless any hard_fail is triggered, in which
case the score is zero regardless of criteria met (see eval-spec.md).

Usage:
    python score_objective.py grade --task tasks/research.yaml --runs-dir runs \
        --grader-provider groq --grader-model llama-3.3-70b-versatile

    python score_objective.py hand-sample --scores-dir scores --fraction 0.15
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import requests
import yaml

from runner import load_providers, load_tasks, call_model, safe_slug

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GRADER_INSTRUCTIONS = """You are grading one model's answer against a fixed \
checklist. Each criterion is binary: met or not met. Do not be lenient \
because the answer is fluent; do not penalize brevity if the criterion is \
still satisfied.

Task prompt given to the model:
---
{prompt}
---

Model's answer:
---
{answer}
---

Criteria (grade each strictly):
{criteria_block}

Hard-fail conditions (if ANY applies, the whole answer fails regardless of \
criteria above):
{hard_fail_block}

Respond with ONLY a JSON object, no prose, no markdown fences, in exactly \
this shape:
{{"criteria_met": {{"c1": true, "c2": false, ...}}, "hard_fail_triggered": \
false, "hard_fail_reason": null}}
"""


def build_grading_prompt(task, answer):
    criteria_block = "\n".join(f"- {c['id']}: {c['text']}" for c in task["criteria"])
    hard_fail_block = "\n".join(f"- {h}" for h in task.get("hard_fail", [])) or "(none)"
    return GRADER_INSTRUCTIONS.format(
        prompt=task["prompt"], answer=answer,
        criteria_block=criteria_block, hard_fail_block=hard_fail_block,
    )


def extract_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object found in grader output: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def compute_score(task, verdict):
    if verdict.get("hard_fail_triggered"):
        return 0.0, verdict.get("hard_fail_reason")
    met = verdict.get("criteria_met", {})
    total = len(task["criteria"])
    if total == 0:
        return 0.0, "task has no criteria"
    n_met = sum(1 for c in task["criteria"] if met.get(c["id"]) is True)
    return n_met / total, None


def cmd_grade(args):
    providers = load_providers(args.providers_file)
    if args.grader_provider not in providers:
        sys.exit(f"Unknown grader provider '{args.grader_provider}'")
    cfg = providers[args.grader_provider]
    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        sys.exit(f"{cfg['api_key_env']} is not set; can't run the grader model")

    tasks = {t["id"]: t for t in load_tasks(args.task)}
    runs_dir = Path(args.runs_dir)
    scores_dir = Path(args.scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)

    transcript_files = [
        f for f in runs_dir.glob("*.json")
        if f.stem.split("__")[0] in tasks
    ]
    if not transcript_files:
        sys.exit(f"No transcripts in {runs_dir} match task ids from {args.task}")

    for f in transcript_files:
        with open(f, encoding="utf-8") as fh:
            transcript = json.load(fh)
        if "error" in transcript:
            print(f"skip {f.name}: run had an error, nothing to grade")
            continue
        task = tasks[transcript["task_id"]]
        answer = transcript["exchanges"][0]["assistant"]

        grading_prompt = build_grading_prompt(task, answer)
        # Grading prompts are longer than a normal conversational turn (full
        # answer + criteria + instructions), so gpt-oss models can burn a small
        # token budget entirely on hidden reasoning and return empty content
        # even with reasoning_effort=low. Give the grader room; the verdict
        # itself is a short JSON object either way.
        reply = call_model(cfg["base_url"], api_key, args.grader_model,
                           [{"role": "user", "content": grading_prompt}],
                           max_tokens=600)
        raw = reply["content"]
        if reply.get("finish_reason") == "length":
            # A verdict cut off mid-JSON parses as a failure or, worse, as a
            # partial criteria dict that scores lower than the answer deserves.
            print(f"SKIP {f.name}: grader verdict hit the token cap and is "
                  f"incomplete", file=sys.stderr)
            continue
        try:
            verdict = extract_json(raw)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"FAILED to parse grader output for {f.name}: {e}", file=sys.stderr)
            continue

        score, hard_fail_reason = compute_score(task, verdict)
        out = {
            "task_id": task["id"],
            "provider": transcript["provider"],
            "model": transcript["model"],
            "run_index": transcript.get("run_index", 0),
            "score": score,
            "criteria_met": verdict.get("criteria_met", {}),
            "hard_fail_triggered": bool(verdict.get("hard_fail_triggered")),
            "hard_fail_reason": hard_fail_reason,
            "grader_provider": args.grader_provider,
            "grader_model": args.grader_model,
            "source_transcript": f.name,
        }
        out_name = f"{task['id']}__{transcript['provider']}_{safe_slug(transcript['model'])}__run{transcript.get('run_index', 0)}.json"
        with open(scores_dir / out_name, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        print(f"{f.name}: score {score:.2f}" + (f" (hard fail: {hard_fail_reason})" if hard_fail_reason else ""))


def cmd_hand_sample(args):
    scores_dir = Path(args.scores_dir)
    all_scores = list(scores_dir.glob("*.json"))
    all_scores = [f for f in all_scores if f.name != ".gitkeep"]
    if not all_scores:
        sys.exit(f"No scored files in {scores_dir}")

    n = max(1, round(len(all_scores) * args.fraction))
    sample = random.sample(all_scores, min(n, len(all_scores)))
    print(f"Hand-grading {len(sample)}/{len(all_scores)} scored transcripts "
          f"({args.fraction:.0%} target).\n")

    disagreements = 0
    for f in sample:
        with open(f, encoding="utf-8") as fh:
            scored = json.load(fh)
        print(f"--- {f.name} ---")
        print(f"model-graded score: {scored['score']:.2f}")
        print(f"source transcript: {scored['source_transcript']}")
        answer = input("Your score (0-1, or 'agree' to accept the model grade): ").strip()
        if answer.lower() != "agree":
            try:
                hand_score = float(answer)
            except ValueError:
                print("not a number, treating as disagreement")
                hand_score = None
            if hand_score is None or abs(hand_score - scored["score"]) > 1e-6:
                disagreements += 1

    rate = disagreements / len(sample)
    print(f"\nDisagreement rate: {disagreements}/{len(sample)} = {rate:.0%}")
    if rate > 0.20:
        print("Above 20% — per eval-spec.md, the automated scores aren't "
              "usable yet. Tighten the rubric before trusting them.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grade")
    g.add_argument("--task", required=True)
    g.add_argument("--runs-dir", default="runs")
    g.add_argument("--scores-dir", default="scores")
    g.add_argument("--providers-file", default="providers.yaml")
    g.add_argument("--grader-provider", required=True)
    g.add_argument("--grader-model", required=True)
    g.set_defaults(func=cmd_grade)

    h = sub.add_parser("hand-sample")
    h.add_argument("--scores-dir", default="scores")
    h.add_argument("--fraction", type=float, default=0.15)
    h.set_defaults(func=cmd_hand_sample)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
