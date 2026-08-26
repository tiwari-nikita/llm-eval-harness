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
providers.yaml    model/endpoint config for runner.py
runner.py         sends task turns to models, saves transcripts
score_objective.py  model-graded + hand-sample grading for research/documents
vote_pairwise.py     blind pairwise voting CLI for advisory/personal
chat_vote.py         same protocol, driven from a chat session instead
router.py            ranks models per category from /scores + /preference
catalogue.py         live catalogue of free model access across providers
FREE_MODELS.md       generated export of the above; regenerate, don't trust
```

## Setup

```
pip install -r requirements.txt
```

Run the test suite (mocks all network calls, no API key needed):

```
python -m pytest
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

# resume a run that a per-day quota cut short, without re-spending tokens
python runner.py --task tasks/personal.yaml --providers groq --skip-existing

# grade an objective run, then hand-check a sample of the grades
python score_objective.py grade --task tasks/research.yaml \
    --grader-provider groq --grader-model openai/gpt-oss-120b
python score_objective.py hand-sample --fraction 0.15

# blind pairwise vote between two advisory transcripts
python vote_pairwise.py --transcripts runs/advisory_001__groq_A.json runs/advisory_001__openrouter_B.json \
    --task-file tasks/advisory.yaml

# ...or take the same vote in a chat session, when a terminal prompt is the
# thing stopping the votes from happening
python chat_vote.py show   --task-id advisory_001 --task-file tasks/advisory.yaml
python chat_vote.py record --task-id advisory_001 --votes a,b,tie,a,a

# what is free right now, fetched rather than remembered
python catalogue.py refresh
python catalogue.py list --free-only
python catalogue.py export -o FREE_MODELS.md
python catalogue.py providers-diff     # is providers.yaml still accurate?

# check whether the OpenRouter models in providers.yaml are still free, then rank
python router.py check-catalogue
python router.py rank --category advisory
python router.py rank --category research
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
- **Replies are length-constrained, and that changes what is measured.**
  `advisory.yaml` and `personal.yaml` send a system prompt asking for under
  350 words. This is not cosmetic: it means the eval measures the best
  *concise* advice a model gives, not its best advice at any length. The
  constraint exists because a 5-turn script resends its whole history each
  turn, so on an 8000 TPM cap the affordable ceiling is around 1500 tokens
  per reply. Remove the `system:` key from both task files if you would
  rather measure something else — but remove it from both, or the comparison
  is meaningless.
- **The daily cap is the real budget, and it is a rolling window.** Groq's
  free tier meters 200k tokens/day for `gpt-oss-120b`, and one full
  24-transcript run very nearly exhausts it. That makes `--n-runs 2+`
  effectively impossible on this tier, so every number here is n=1 per
  model per task with no variance estimate. The window rolls rather than
  resetting at midnight: headroom returns continuously as old usage ages
  out, and admission is decided against instantaneous capacity — a
  7000-token request was observed succeeding seconds after a 3000-token one
  was rejected. `--skip-existing` exists so a run killed by this can be
  resumed without re-spending the budget.
- **Personal category is local-only.** `preference/personal_*` and
  `/prompts` never get committed, by `.gitignore`, because they're private
  conversations, not test fixtures.
- **Router's "free and within quota" is partial.** OpenRouter publishes a
  live per-model free/paid flag (`router.py check-catalogue` checks it for
  real), so that half is genuinely verified against a maintained catalogue.
  Groq and Google AI Studio don't expose an equivalent per-model endpoint —
  their free tier is an account-level rate limit — so the router trusts
  `providers.yaml` for those two rather than faking a live check. There is
  also no live per-key quota/usage API wired up anywhere yet; the router
  reports quota as "unknown" rather than pretending otherwise.
- **The catalogue can only verify what providers publish.** `catalogue.py`
  fetches model lists rather than remembering them, but free-ness is only
  machine-checkable where a provider publishes per-model pricing
  (OpenRouter, SambaNova) or a free flag (HuggingFace). NVIDIA lists 95
  models with no pricing field at all; Groq, Google, Cerebras, Mistral and
  Together meter at the account level. Those are reported as *unknown*, not
  as free and not as paid — the three-valued distinction is deliberate,
  since "costs money" and "the endpoint didn't say" lead to different
  decisions. Product tiers and local runtimes are catalogued too, but
  nothing there is API-callable by `runner.py` except a local runtime you
  are actually running.
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

### Live runs so far (Groq, 2026-08-26)

The only provider with a live key so far is Groq (`openai/gpt-oss-120b` vs
`openai/gpt-oss-20b`). 24 transcripts, all healthy, 3% of replies truncated.

| category  | gpt-oss-120b | gpt-oss-20b | n per model |
|-----------|-------------:|------------:|------------:|
| research  | 0.50         | 0.50        | 2           |
| documents | 0.88         | 0.94        | 4           |

Advisory and personal have 6 transcripts each, ready for blind pairwise
voting; **none cast yet** — that step needs a human, not the harness.

Four things about these numbers matter more than the numbers:

- **The previous ranking was a measurement artifact.** The table here used to
  read research 0.58/0.38 and documents 0.88/0.62, apparently a clear win for
  the larger model. Those scores were computed on replies the harness had
  truncated at 250 tokens — 82% of every saved reply ended mid-sentence. Rerun
  untruncated, research ties and documents *reverses*. Nothing about the
  models changed; the instrument did. Treat any result here as provisional
  until you know what the harness was doing to the inputs.
- **Both models hard-failed `research_001` for fabricated citations**, which
  is what zeroes that row. That is the single most interesting result so far
  and the least trustworthy: the grader is asserting that specific papers do
  not exist, and "does this paper exist" is exactly the judgment a language
  model is worst at. It could equally be a false positive. Hand-check this
  one first — it is the highest-leverage item in the whole score set.
- **The model grader is not fully deterministic.** Re-running it on an
  unchanged transcript once flipped a criterion and moved the score from 1.00
  to 0.75 with no other input changed. This is why eval-spec.md requires
  hand-checking a random 15% and reporting the disagreement rate rather than
  trusting one automated pass. That hand-check still hasn't been run.
- **The grader is the same model family as both models it grades**, since
  Groq is the only provider with a key. Take all of this as pipeline
  validation, not a ranking, until a second provider's key exists. Google AI
  Studio is the useful one to add: independent family, so the grader stops
  being related to the graded, and a separate daily budget.

## Grading

Objective sets (`research.yaml`, `documents.yaml`): binary checklist per
`eval-spec.md`, model-graded first pass, then hand-grade a random 15% and
report the disagreement rate. Above ~20% disagreement the automated scores
aren't usable.

Advisory: blind pairwise only, on `held_position`, `specificity`,
`context_retention`, `honesty`, `overall`. Model identities hidden until
the vote is recorded; log the date, since preferences drift.
