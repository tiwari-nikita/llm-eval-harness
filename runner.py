#!/usr/bin/env python3
"""
Send task turns to models over an OpenAI-compatible chat endpoint and save
raw transcripts under /runs.

One HTTP client covers Groq, OpenRouter, and Google AI Studio, since all
three speak the same /chat/completions shape (see providers.yaml).

Usage:
    python runner.py --task tasks/advisory.yaml --providers groq
    python runner.py --task tasks/research.yaml --providers groq,openrouter --n-runs 2

Transcripts land in /runs/<task_id>__<provider>_<model>__run<n>.json and
are gitignored. Nothing here scores anything; that's a separate pass.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RUNS_DIR = Path(__file__).parent / "runs"

# Groq's free tier meters tokens per minute account-wide; 8000 is what this
# account reports. Override with --tpm for a provider with a different cap.
DEFAULT_TPM = 8000
DEFAULT_MAX_TOKENS = 1200


def load_providers(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tasks(path):
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict):
        return data.get("tasks", [])
    return data or []


def safe_slug(text):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)


class TokenBudget:
    """Sliding-window limiter for an account-level tokens-per-minute cap.

    Groq's free tier meters TPM, not requests, and a multi-turn task resends
    the whole history every turn — so turn 5 of a conversation costs several
    times what turn 1 did. A fixed sleep between calls therefore either
    over-waits at the start or blows through the cap at the end. This reserves
    each call's estimated cost up front and settles it against the usage the
    response actually reports.
    """

    def __init__(self, tpm, window=60.0):
        self.tpm = tpm
        self.window = window
        self.events = []  # [(monotonic_ts, tokens)]

    def _prune(self, now):
        cutoff = now - self.window
        self.events = [(t, n) for t, n in self.events if t > cutoff]

    def reserve(self, tokens):
        while True:
            now = time.monotonic()
            self._prune(now)
            used = sum(n for _, n in self.events)
            # the `not self.events` arm keeps a single oversized call from
            # deadlocking against a cap it can never fit under on its own
            if used + tokens <= self.tpm or not self.events:
                self.events.append((now, tokens))
                return
            wait = max(0.0, self.events[0][0] + self.window - now) + 0.5
            print("    TPM {}/{}, need {}: waiting {:.0f}s".format(
                used, self.tpm, tokens, wait), file=sys.stderr)
            time.sleep(wait)

    def settle(self, actual):
        """Replace the last reservation's estimate with real reported usage."""
        if self.events and actual:
            ts, _ = self.events[-1]
            self.events[-1] = (ts, actual)


def estimate_tokens(messages, max_tokens):
    """Rough pre-flight cost of a call: ~4 chars per token, plus the output cap."""
    chars = sum(len(m.get("content") or "") for m in messages)
    return chars // 4 + max_tokens


def call_model(base_url, api_key, model, messages, timeout=120, max_retries=4,
               max_tokens=DEFAULT_MAX_TOKENS, budget=None):
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    # max_tokens caps reply length. It has to be generous enough that the
    # model, not the harness, decides where an answer ends: at the old value
    # of 250, 82% of saved replies stopped mid-sentence, which silently turned
    # every comparison into "who front-loads best" rather than "who answers
    # best". Pacing against the TPM cap (see TokenBudget) is what makes a
    # roomy cap affordable on a free tier.
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if "gpt-oss" in model:
        # gpt-oss models spend part of the max_tokens budget on a hidden
        # <reasoning> trace before emitting any visible content; at a small
        # budget that leaves content empty. reasoning_effort="low" fixes that
        # and keeps free-tier token spend down. Other providers reject unknown
        # fields differently, so keep it scoped to gpt-oss only.
        payload["reasoning_effort"] = "low"

    if budget is not None:
        budget.reserve(estimate_tokens(messages, max_tokens))

    for attempt in range(max_retries):
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = min(2 ** attempt * 2, 30)
            print("    [{}] retrying in {}s...".format(resp.status_code, wait),
                  file=sys.stderr)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        body = resp.json()
        choice = body["choices"][0]
        usage = body.get("usage") or {}
        if budget is not None:
            budget.settle(usage.get("total_tokens"))
        return {
            "content": choice["message"].get("content") or "",
            # "length" means the harness cut the reply off, not the model.
            # Saving it is the only way a later reader can tell those apart.
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

    resp.raise_for_status()


def run_one(task, provider_name, provider_cfg, model, api_key, run_index,
            max_tokens=DEFAULT_MAX_TOKENS, budget=None):
    is_multiturn = "turns" in task
    turns = task["turns"] if is_multiturn else [task["prompt"]]

    messages = []
    exchanges = []
    for turn_text in turns:
        messages.append({"role": "user", "content": turn_text})
        reply = call_model(provider_cfg["base_url"], api_key, model, messages,
                           max_tokens=max_tokens, budget=budget)
        messages.append({"role": "assistant", "content": reply["content"]})
        exchanges.append({
            "user": turn_text,
            "assistant": reply["content"],
            "finish_reason": reply["finish_reason"],
            "completion_tokens": reply["completion_tokens"],
            "truncated": reply["finish_reason"] == "length",
        })
        time.sleep(1)  # politeness gap; the TPM budget does the real pacing

    n_cut = sum(1 for e in exchanges if e["truncated"])
    if n_cut:
        print("    WARNING: {}/{} replies hit the {}-token cap and are "
              "truncated".format(n_cut, len(exchanges), max_tokens), file=sys.stderr)

    return {
        "task_id": task["id"],
        "provider": provider_name,
        "model": model,
        "run_index": run_index,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "multiturn": is_multiturn,
        "max_tokens": max_tokens,
        "n_truncated": n_cut,
        "exchanges": exchanges,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="path to a task yaml file")
    ap.add_argument("--providers", required=True,
                     help="comma-separated provider names from providers.yaml")
    ap.add_argument("--providers-file", default="providers.yaml")
    ap.add_argument("--n-runs", type=int, default=1,
                     help="repeat each task this many times per model")
    ap.add_argument("--out-dir", default=str(RUNS_DIR))
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                     help="per-reply output cap; set too low and the harness, "
                          "not the model, decides where answers end")
    ap.add_argument("--tpm", type=int, default=DEFAULT_TPM,
                     help="account-wide tokens-per-minute cap to pace against")
    args = ap.parse_args()

    all_providers = load_providers(args.providers_file)
    tasks = load_tasks(args.task)
    if not tasks:
        sys.exit(f"No tasks found in {args.task}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requested = [p.strip() for p in args.providers.split(",") if p.strip()]
    missing_keys = []
    jobs = []
    for pname in requested:
        if pname not in all_providers:
            sys.exit(f"Unknown provider '{pname}', check {args.providers_file}")
        cfg = all_providers[pname]
        import os
        api_key = os.environ.get(cfg["api_key_env"])
        if not api_key:
            missing_keys.append(cfg["api_key_env"])
            continue
        for model in cfg["models"]:
            jobs.append((pname, cfg, model, api_key))

    if missing_keys:
        print("Skipping providers with no API key set: "
              + ", ".join(sorted(set(missing_keys))), file=sys.stderr)
    if not jobs:
        sys.exit("No runnable (provider, model) pairs. Set at least one "
                  "provider's API key env var and try again.")

    budget = TokenBudget(args.tpm)
    total = len(tasks) * len(jobs) * args.n_runs
    done = 0
    truncated_total = 0
    for task in tasks:
        for pname, cfg, model, api_key in jobs:
            for run_index in range(args.n_runs):
                done += 1
                print(f"[{done}/{total}] {task['id']} x {pname}/{model} run {run_index}")
                try:
                    result = run_one(task, pname, cfg, model, api_key, run_index,
                                     max_tokens=args.max_tokens, budget=budget)
                    truncated_total += result["n_truncated"]
                except Exception as e:
                    print(f"    FAILED: {e}", file=sys.stderr)
                    result = {
                        "task_id": task["id"], "provider": pname, "model": model,
                        "run_index": run_index,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "error": str(e),
                    }
                fname = f"{task['id']}__{pname}_{safe_slug(model)}__run{run_index}.json"
                with open(out_dir / fname, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {done} transcripts to {out_dir}")
    if truncated_total:
        print(f"WARNING: {truncated_total} replies were truncated at the "
              f"{args.max_tokens}-token cap. A comparison over truncated "
              f"replies measures front-loading, not answer quality — raise "
              f"--max-tokens and re-run before scoring or voting.", file=sys.stderr)


if __name__ == "__main__":
    main()
