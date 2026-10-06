#!/usr/bin/env python3
"""
Put uncertainty and grader agreement on the saved scores. Reads score files
only, so it costs no tokens.

    # per-model means with intervals, and paired model-vs-model comparisons
    python report_stats.py scores

    # ...plus agreement between two graders who scored the same transcripts
    python report_stats.py scores --second scores_gemini

    # ...plus agreement with your own hand grades (see score_objective.py hand-sample)
    python report_stats.py scores --hand scores_hand/hand_grades.json

    # write the report to a file as well as printing it
    python report_stats.py scores --second scores_gemini --out stats_report.md

Methods, in one place so the numbers can be checked:
  - A model's mean score gets a 95% percentile-bootstrap interval over its
    transcripts. At the sizes here (2-4 per category) that interval is too
    narrow, so read it as the least uncertainty there is.
  - Two models are compared task by task (paired), with an exact sign test for
    the verdict. With fewer than 6 tasks where they differ, no verdict can
    reach p < 0.05, and the report says so instead of naming a winner.
  - "Flips to erase lead" is the fewest single-criterion grading changes that
    would tie or reverse the ranking. Compare it with how often two graders
    actually disagree.
  - Rates (hard fails, disagreements) get 95% Wilson intervals.
  - Grader agreement on criteria and hard fails is Cohen's kappa, which
    discounts the agreement two graders would reach by chance.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

from stats import (MIN_TASKS_FOR_SIGN_TEST, bootstrap_mean_ci, cohen_kappa,
                   flips_to_erase_lead, paired_difference, wilson_interval)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_scores(scores_dir):
    """Score files in the top level of scores_dir (archives in subfolders are
    skipped), keyed by file name."""
    out = {}
    for f in sorted(Path(scores_dir).glob("*.json")):
        with open(f, encoding="utf-8") as fh:
            out[f.name] = json.load(fh)
    return out


def category(task_id):
    return task_id.rsplit("_", 1)[0]


def model_labels(records):
    """Short model names, falling back to provider/model if two providers
    serve models with the same short name."""
    full = {(r["provider"], r["model"]) for r in records}
    short = {}
    for provider, model in full:
        short.setdefault(model.split("/")[-1], []).append((provider, model))
    labels = {}
    for name, owners in short.items():
        for provider, model in owners:
            labels[(provider, model)] = name if len(owners) == 1 else f"{provider}/{model}"
    return labels


def met_total(record):
    """(criteria met, criteria total) as the score counts them: a hard fail
    scores 0, so it counts as nothing met."""
    crit = record.get("criteria_met") or {}
    total = len(crit)
    met = 0 if record.get("hard_fail_triggered") else sum(bool(v) for v in crit.values())
    return met, total


def fmt(x):
    return f"{x:.2f}"


def fmt_interval(lo, hi, n):
    if lo is None:
        return f"n/a (n={n})"
    return f"{fmt(lo)} to {fmt(hi)}"


def fmt_rate(k, n):
    lo, hi = wilson_interval(k, n)
    return f"{k}/{n} ({lo:.0%} to {hi:.0%})"


def graders(scores):
    return ", ".join(sorted({r.get("grader_model") or "unknown" for r in scores.values()}))


def by_category_and_model(scores):
    """{category: {model label: {task_id: record}}}"""
    labels = model_labels(scores.values())
    out = {}
    for r in scores.values():
        label = labels[(r["provider"], r["model"])]
        out.setdefault(category(r["task_id"]), {}).setdefault(label, {})[r["task_id"]] = r
    return out


def score_section(scores, seed, n_boot):
    lines = [f"Grader: {graders(scores)}", ""]
    for cat, models in sorted(by_category_and_model(scores).items()):
        lines += [f"## {cat}", "",
                  "| model | n | mean | 95% interval | hard fails |",
                  "|---|---:|---:|---|---|"]
        for label, tasks in sorted(models.items()):
            vals = [r["score"] for r in tasks.values()]
            m, lo, hi = bootstrap_mean_ci(vals, n_boot=n_boot, seed=seed)
            hf = sum(bool(r.get("hard_fail_triggered")) for r in tasks.values())
            lines.append(f"| {label} | {len(vals)} | {fmt(m)} | {fmt_interval(lo, hi, len(vals))} "
                         f"| {fmt_rate(hf, len(vals))} |")
        lines.append("")
        pairs = list(itertools.combinations(sorted(models), 2))
        if pairs:
            lines += ["Paired by task (A minus B):", "",
                      "| A vs B | tasks | A - B | 95% interval | A wins-losses-ties | sign-test p "
                      "| verdict | flips to erase lead |",
                      "|---|---:|---:|---|---|---:|---|---:|"]
            for a, b in pairs:
                sa = {t: r["score"] for t, r in models[a].items()}
                sb = {t: r["score"] for t, r in models[b].items()}
                if not set(sa) & set(sb):
                    continue
                res = paired_difference(sa, sb, n_boot=n_boot, seed=seed)
                ma = {t: met_total(r) for t, r in models[a].items()}
                mb = {t: met_total(r) for t, r in models[b].items()}
                if res["diff"] >= 0:
                    flips = flips_to_erase_lead(ma, mb)
                else:
                    flips = flips_to_erase_lead(mb, ma)
                verdict = res["verdict"]
                if verdict == "not distinguishable" and res["wins"] + res["losses"] < MIN_TASKS_FOR_SIGN_TEST:
                    verdict += " (too few tasks for any verdict)"
                lines.append(f"| {a} vs {b} | {res['n']} | {res['diff']:+.2f} "
                             f"| {fmt_interval(res['lo'], res['hi'], res['n'])} "
                             f"| {res['wins']}-{res['losses']}-{res['ties']} | {res['p']:.3f} "
                             f"| {verdict} | {flips} |")
            lines.append("")
    return lines


def agreement_section(first, second):
    shared = sorted(set(first) & set(second))
    lines = ["# Grader agreement", "",
             f"First grader: {graders({k: first[k] for k in shared})}. "
             f"Second grader: {graders({k: second[k] for k in shared})}. "
             f"{len(shared)} transcripts scored by both.", ""]
    if not shared:
        return lines + ["No transcripts in common.", ""]
    groups = {}
    for name in shared:
        groups.setdefault(category(first[name]["task_id"]), []).append(name)
    groups_all = sorted(groups.items()) + [("overall", shared)]
    lines += ["| category | transcripts | scores differ | criterion judgments | raw agreement "
              "| kappa (criteria) | kappa (hard fail) |",
              "|---|---:|---|---:|---:|---:|---:|"]
    for cat, names in groups_all:
        differ = sum(abs(first[n]["score"] - second[n]["score"]) > 1e-9 for n in names)
        crit_pairs = []
        for n in names:
            ca, cb = first[n].get("criteria_met") or {}, second[n].get("criteria_met") or {}
            crit_pairs += [(ca[c], cb[c]) for c in sorted(set(ca) & set(cb))]
        hf_pairs = [(first[n].get("hard_fail_triggered"), second[n].get("hard_fail_triggered")) for n in names]
        ck = cohen_kappa(crit_pairs) if crit_pairs else None
        hk = cohen_kappa(hf_pairs)
        kc = "n/a" if ck is None or ck["kappa"] is None else fmt(ck["kappa"])
        kh = "n/a" if hk["kappa"] is None else fmt(hk["kappa"])
        raw = "n/a" if ck is None else f"{ck['agreement']:.0%}"
        lines.append(f"| {cat} | {len(names)} | {fmt_rate(differ, len(names))} | "
                     f"{0 if ck is None else ck['n']} | {raw} | {kc} | {kh} |")
    lines += ["", "kappa is n/a when both graders gave the same label to everything, "
              "so there is no variation to agree on beyond chance.", ""]

    lines += ["Does the ranking depend on the grader? (means over the transcripts both graded)", "",
              "| category | model | first grader | second grader |", "|---|---|---:|---:|"]
    labels = model_labels([first[n] for n in shared])
    flips = []
    for cat, names in sorted(groups.items()):
        per_model = {}
        for n in names:
            label = labels[(first[n]["provider"], first[n]["model"])]
            per_model.setdefault(label, ([], []))
            per_model[label][0].append(first[n]["score"])
            per_model[label][1].append(second[n]["score"])
        means = {m: (sum(a) / len(a), sum(b) / len(b)) for m, (a, b) in per_model.items()}
        for m, (x, y) in sorted(means.items()):
            lines.append(f"| {cat} | {m} | {fmt(x)} | {fmt(y)} |")
        if len(means) >= 2:
            top1 = max(means.values(), key=lambda v: v[0])[0]
            top2 = max(means.values(), key=lambda v: v[1])[1]
            lead1 = sorted(m for m, v in means.items() if v[0] == top1)
            lead2 = sorted(m for m, v in means.items() if v[1] == top2)
            if lead1 != lead2:
                flips.append(f"{cat}: first grader puts {' = '.join(lead1)} on top, "
                             f"second grader puts {' = '.join(lead2)} on top")
    lines.append("")
    lines += (["Ranking changes with the grader:", *[f"- {f}" for f in flips]] if flips
              else ["The top model is the same under both graders in every category."])
    return lines + [""]


def hand_section(path):
    lines = ["# Model grades vs your hand grades", ""]
    if not Path(path).exists():
        return lines + [f"No hand grades recorded yet ({path} doesn't exist). "
                        "Run `score_objective.py hand-sample` first.", ""]
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)
    if not records:
        return lines + ["No hand grades recorded yet.", ""]
    differ = sum(not r["agree"] for r in records)
    lines += [f"Disagreement: {fmt_rate(differ, len(records))} of hand-checked transcripts. "
              "eval-spec.md treats anything above 20% as unusable; compare that line with the "
              "interval, not just the point estimate.", ""]
    return lines


def build_report(scores_dir, second=None, hand=None, seed=0, n_boot=10_000):
    first = load_scores(scores_dir)
    if not first:
        sys.exit(f"No score files in {scores_dir}")
    lines = ["# Score statistics", "",
             f"Scores: `{scores_dir}`. 95% intervals; bootstrap uses {n_boot} resamples, seed {seed}. "
             "With 2-4 transcripts per model the bootstrap interval is too narrow; read it as a floor "
             "on the uncertainty.", ""]
    lines += score_section(first, seed, n_boot)
    if second:
        lines += agreement_section(first, load_scores(second))
    if hand:
        lines += hand_section(hand)
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scores_dir")
    p.add_argument("--second", help="a second grader's score directory, same file names")
    p.add_argument("--hand", help="hand grades JSON written by score_objective.py hand-sample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--boot", type=int, default=10_000, help="bootstrap resamples")
    p.add_argument("--out", help="also write the report to this file")
    args = p.parse_args()
    report = build_report(args.scores_dir, args.second, args.hand, args.seed, args.boot)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
