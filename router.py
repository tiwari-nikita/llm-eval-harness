#!/usr/bin/env python3
"""
Rank models per category from /scores and /preference, filtered against a
live free-tier catalogue where one actually exists.

Two subcommands:
  rank              print a ranked model list for one category
  check-catalogue   validate providers.yaml's OpenRouter models against
                    OpenRouter's live /models endpoint (pricing == 0)

Honesty about what "filtered by free and quota" can actually mean here:
OpenRouter publishes a live, queryable list of which model IDs are
currently free (https://openrouter.ai/api/v1/models, pricing.prompt == 0).
Groq and Google AI Studio don't expose an equivalent "is this specific
model free right now" endpoint — their free tier is an account-level rate
limit, not a per-model flag. So for those two providers this script trusts
providers.yaml (a human already chose free-tier models) rather than
pretending to verify it live. That's a real gap, not an oversight, and gets
printed as a warning rather than hidden.

There's no live per-key quota/usage API wired up either. Quota is reported
as "unknown" until something concrete backs it — a router that fakes quota
awareness is worse than one that admits it doesn't have it.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CATALOGUE_CACHE = Path(__file__).parent / ".openrouter_catalogue_cache.json"
CATALOGUE_MAX_AGE_S = 24 * 3600


def category_of(task_id):
    return task_id.split("_")[0]


def load_scores(scores_dir):
    out = []
    for f in Path(scores_dir).glob("*.json"):
        if f.name == ".gitkeep":
            continue
        with open(f, encoding="utf-8") as fh:
            out.append(json.load(fh))
    return out


def load_preference(preference_dir, include_personal=False):
    out = []
    for f in Path(preference_dir).glob("*.json"):
        if f.name.startswith("personal_") and not include_personal:
            continue
        with open(f, encoding="utf-8") as fh:
            out.append(json.load(fh))
    return out


def rank_objective(category, scores):
    agg = defaultdict(lambda: {"total": 0.0, "n": 0})
    for s in scores:
        if category_of(s["task_id"]) != category:
            continue
        key = (s["provider"], s["model"])
        agg[key]["total"] += s["score"]
        agg[key]["n"] += 1
    rows = [
        {"provider": p, "model": m, "mean_score": v["total"] / v["n"], "n_tasks": v["n"]}
        for (p, m), v in agg.items()
    ]
    return sorted(rows, key=lambda r: r["mean_score"], reverse=True)


def rank_pairwise(category, votes, dimension="overall"):
    record = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0})
    for v in votes:
        if category_of(v["task_id"]) != category:
            continue
        outcome = v["votes"].get(dimension)
        a, b = v["model_a"], v["model_b"]
        if outcome == "a":
            record[a]["wins"] += 1
            record[b]["losses"] += 1
        elif outcome == "b":
            record[b]["wins"] += 1
            record[a]["losses"] += 1
        elif outcome == "tie":
            record[a]["ties"] += 1
            record[b]["ties"] += 1

    rows = []
    for model, r in record.items():
        decisive = r["wins"] + r["losses"]
        win_rate = r["wins"] / decisive if decisive else None
        rows.append({
            "model": model, "wins": r["wins"], "losses": r["losses"],
            "ties": r["ties"], "win_rate": win_rate,
        })
    rows.sort(key=lambda r: (r["win_rate"] is not None, r["win_rate"]), reverse=True)
    return rows


def fetch_live_free_openrouter_models(force=False):
    if not force and CATALOGUE_CACHE.exists():
        age = time.time() - CATALOGUE_CACHE.stat().st_mtime
        if age < CATALOGUE_MAX_AGE_S:
            with open(CATALOGUE_CACHE, encoding="utf-8") as f:
                return set(json.load(f)), age

    try:
        resp = requests.get("https://openrouter.ai/api/v1/models", timeout=15)
        resp.raise_for_status()
        models = resp.json()["data"]
        free_ids = [
            m["id"] for m in models
            if m.get("pricing", {}).get("prompt") == "0"
            and m.get("pricing", {}).get("completion") == "0"
        ]
        with open(CATALOGUE_CACHE, "w", encoding="utf-8") as f:
            json.dump(free_ids, f)
        return set(free_ids), 0
    except requests.RequestException as e:
        if CATALOGUE_CACHE.exists():
            print(f"live fetch failed ({e}), falling back to stale cache", file=sys.stderr)
            with open(CATALOGUE_CACHE, encoding="utf-8") as f:
                return set(json.load(f)), time.time() - CATALOGUE_CACHE.stat().st_mtime
        raise


def cmd_check_catalogue(args):
    providers = yaml.safe_load(open(args.providers_file, encoding="utf-8"))
    or_cfg = providers.get("openrouter")
    if not or_cfg:
        print("no 'openrouter' entry in providers.yaml, nothing to check")
        return

    free_ids, age = fetch_live_free_openrouter_models(force=args.refresh)
    print(f"live OpenRouter free-model catalogue ({len(free_ids)} models, "
          f"cache age {age/3600:.1f}h)\n")

    for model in or_cfg["models"]:
        status = "OK, currently free" if model in free_ids else "STALE — not in the current free list"
        print(f"  {model}: {status}")

    for pname in providers:
        if pname != "openrouter":
            print(f"\n[{pname}] no live per-model free-tier catalogue exists; "
                  f"trusting providers.yaml as-is (not independently verified)")


def cmd_rank(args):
    scores = load_scores(args.scores_dir)
    votes = load_preference(args.preference_dir, include_personal=args.category == "personal")

    print(f"category: {args.category}\n")
    if args.category in ("research", "documents"):
        rows = rank_objective(args.category, scores)
        if not rows:
            print("(no scores yet for this category)")
        for r in rows:
            print(f"  {r['mean_score']:.2f}  {r['provider']}/{r['model']}  (n={r['n_tasks']})")
    else:
        rows = rank_pairwise(args.category, votes)
        if not rows:
            print("(no preference votes yet for this category)")
        for r in rows:
            wr = f"{r['win_rate']:.0%}" if r["win_rate"] is not None else "n/a"
            print(f"  win_rate={wr}  {r['model']}  "
                  f"(w={r['wins']} l={r['losses']} t={r['ties']})")

    print("\nquota: unknown — no live per-key usage API is wired up. "
          "Free-tier status for groq/google is trusted from providers.yaml, "
          "not independently verified; run "
          "`python router.py check-catalogue` for the OpenRouter half.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("rank")
    r.add_argument("--category", required=True,
                    choices=["advisory", "research", "documents", "personal"])
    r.add_argument("--scores-dir", default="scores")
    r.add_argument("--preference-dir", default="preference")
    r.set_defaults(func=cmd_rank)

    c = sub.add_parser("check-catalogue")
    c.add_argument("--providers-file", default="providers.yaml")
    c.add_argument("--refresh", action="store_true", help="bypass the 24h cache")
    c.set_defaults(func=cmd_check_catalogue)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
