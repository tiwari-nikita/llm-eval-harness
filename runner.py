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

RUNS_DIR = Path(__file__).parent / "runs"


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


def call_model(base_url, api_key, model, messages, timeout=60, max_retries=4):
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages}

    for attempt in range(max_retries):
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = min(2 ** attempt * 2, 30)
            print(f"    [{resp.status_code}] retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        body = resp.json()
        return body["choices"][0]["message"]["content"]

    resp.raise_for_status()


def run_one(task, provider_name, provider_cfg, model, api_key, run_index):
    is_multiturn = "turns" in task
    turns = task["turns"] if is_multiturn else [task["prompt"]]

    messages = []
    exchanges = []
    for turn_text in turns:
        messages.append({"role": "user", "content": turn_text})
        reply = call_model(provider_cfg["base_url"], api_key, model, messages)
        messages.append({"role": "assistant", "content": reply})
        exchanges.append({"user": turn_text, "assistant": reply})

    return {
        "task_id": task["id"],
        "provider": provider_name,
        "model": model,
        "run_index": run_index,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "multiturn": is_multiturn,
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

    total = len(tasks) * len(jobs) * args.n_runs
    done = 0
    for task in tasks:
        for pname, cfg, model, api_key in jobs:
            for run_index in range(args.n_runs):
                done += 1
                print(f"[{done}/{total}] {task['id']} x {pname}/{model} run {run_index}")
                try:
                    result = run_one(task, pname, cfg, model, api_key, run_index)
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


if __name__ == "__main__":
    main()
