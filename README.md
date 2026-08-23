# Model eval

A personal harness for picking which free-tier model to use for what, based
on this person's actual usage rather than public leaderboards. Full design
rationale is in [eval-spec.md](eval-spec.md).

## Layout

```
/tasks        task files: advisory.yaml (multi-turn), research.yaml, documents.yaml
/prompts      your extracted prompts, gitignored, never committed
/runs         raw transcripts from runner.py, gitignored
/scores       graded objective results
/preference   blind pairwise votes
/router       selection logic reading /scores and /preference
providers.yaml   model/endpoint config for runner.py
runner.py     sends task turns to models, saves transcripts
```

## Setup

```
pip install requests pyyaml datasets pandas pyarrow
```

Set at least one provider's API key as an environment variable (see
providers.yaml for the exact name expected per provider). None of the
configured providers require a card for the tier used, but free-tier model
IDs drift — check the provider's current docs if a model returns 404.

## Running

```
python runner.py --task tasks/advisory.yaml --providers groq
python analyze_corpus.py --hf allenai/WildChat-1M --limit 500
python extract_prompts.py my_export.json -o prompts/mine.jsonl
```

## Known limitations (by design, not oversight)

- **Fixed turns.** Multi-turn advisory scripts send identical text to every
  model regardless of how it replies. This is artificial — a real
  conversation would branch. It's also the only way a comparison between
  models means anything instead of a comparison between two different
  conversations.
- **N of 1.** The blind-pairwise preference data reflects one person's
  judgment on one day. It does not generalize, and a router trained on it
  is supposed to be personal, not a public leaderboard.
- **Personal category is local-only.** `preference/personal_*` and
  `/prompts` never get committed, by `.gitignore`, because they're private
  conversations, not test fixtures.
- **Taxonomy is heuristic.** The regex bucket classifier in `taxonomy.py`
  is wrong on a real, non-trivial share of turns (see below) — good enough
  for a distribution, not for scoring. Hand-check samples per bucket before
  trusting any weighting derived from it.

## What real data has already shown

A 495-conversation pull from `allenai/WildChat-1M` (2026-08-22,
unauthenticated, see `wildchat_500_summary.txt`):

- Mean user turns per conversation: **2.61**. The advisory scripts here run
  5 fixed turns — multi-turn reasoning is genuinely under-represented in
  public chat corpora, which is the actual evidence for weighting advisory
  as the main category rather than an assumption.
- **67.5%** of opening turns didn't match any specific taxonomy bucket
  (fell to `other`). The classifier's keyword rules are English-only and
  narrower than real phrasing diversity — treat any bucket percentage as
  directional, not precise.
- WildChat is public logged chat traffic, which means a lot of jailbreak
  attempts (DAN/AIM-style prompts) and non-English turns the classifier
  doesn't handle. That's a property of this corpus, not necessarily of
  typical usage — don't treat WildChat's distribution as ground truth for
  "how people use models," only as one public reference point.

## Grading

Objective sets (`research.yaml`, `documents.yaml`): binary checklist per
`eval-spec.md`, model-graded first pass, then hand-grade a random 15% and
report the disagreement rate. Above ~20% disagreement the automated scores
aren't usable.

Advisory: blind pairwise only, on `held_position`, `specificity`,
`context_retention`, `honesty`, `overall`. Model identities hidden until
the vote is recorded; log the date, since preferences drift.
