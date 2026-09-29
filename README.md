# LLM Evaluation Harness & Model Router

A personal harness for picking which free-tier model to use for what, based
on my actual usage rather than public leaderboards. Full design
rationale is in [eval-spec.md](eval-spec.md).

## Start here: replay

The rest of this harness ran for weeks without producing an answer to its
one question. The advisory category, which the router weights most heavily,
has zero votes, because one vote meant reading about 4,500 words and making
five judgments. The model grader, the only other signal, turned out to
decide which model won. And the test tasks were mostly about AI research,
which is roughly 2.5% of what the chat history is actually about.

`replay.py` answers the question from the other direction. It takes the
opening prompt of each of your real conversations, sends a sample to two
free models at a time, and asks you to pick between the answers with one
keypress. What comes out is a model card: which model to use for each kind
of thing you actually ask, and how sure that is.

```
python replay.py sample                                  # nothing is sent
#   open prompts/replay/review.html, untick anything, download the approval
python replay.py run --approved approved_<id>.json --dry-run
python replay.py run --approved approved_<id>.json       # builds pick.html
#   open prompts/replay/pick.html:  ← A   → B   ↓ tie   X both bad
python replay.py card picks_<id>.json
```

Privacy is enforced in code, not left to care. `sample` sends nothing.
Prompts that mention a relationship, health, money, a visa, birth details
or contact details start unticked. Ticking a whole group never opts in a
flagged prompt. `run` refuses any prompt or provider the approval doesn't
list, and refuses an approval made for a different sample. Everything lives
under `prompts/replay/`, which is gitignored. The review page names each
provider's data-use terms. Google AI Studio's free tier, for one, may use
prompts to improve Google products.

The pick page holds no model names and never reveals them, not even after a
pick. After a few dozen reveals you would learn each model's house style and
start voting on the brand. `card` is what joins picks back to models.

## Layout

```
/tasks        task files: advisory.yaml (multi-turn), research.yaml, documents.yaml
/prompts      your extracted prompts, gitignored, never committed
/runs         raw transcripts from runner.py, gitignored
/scores       graded objective results
/preference   blind pairwise votes
/router       selection logic reading /scores and /preference
replay.py         real prompts -> blind one-keypress picks -> your model card
replay_review_template.html  privacy review page; nothing is sent until approved
replay_pick_template.html    the pick page; holds no model names
providers.yaml    model/endpoint config for runner.py
runner.py         sends task turns to models, saves transcripts
score_objective.py  model-graded + hand-sample grading for research/documents
vote_pairwise.py     blind pairwise voting CLI for advisory/personal
chat_vote.py         same protocol, driven from a chat session instead
make_grading_bundle.py  builds a standalone HTML page for other people to vote in
grade_template.html     that page's source; generator inlines the transcripts
ingest_votes.py         turns a returned votes file into preference records
verify_citations.py     checks citations against OpenAlex/Crossref, not a model
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

# ...or hand the vote to someone who does not have this repo. Builds one
# self-contained .html that opens from file://, offline, no install; they
# send back a small JSON.
python make_grading_bundle.py --category advisory
python ingest_votes.py votes_advisory_*.json --dry-run
python ingest_votes.py votes_advisory_*.json

# check whether a citation actually exists, rather than asking a model
python verify_citations.py transcript runs/research_001__groq_openai_gpt-oss-120b__run0.json
python verify_citations.py title "Scaling Laws for Reward Model Overoptimization"

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

- **Replay measures short answers to opening prompts.** Every model gets
  "Keep your reply under 150 words", so that a pick takes about a minute
  rather than five. And only the first message of each conversation is
  replayed, since later turns lean on context the replay can't supply.
  So the card ranks each model's best short first answer, not its best
  conversation. Openings with an image or file attached are left out, since
  the models would never see the attachment.
- **Replay's topics are keyword rules, tuned to one chat history.** They're
  better than `taxonomy.py` on this corpus: 18% land in "other", against 88%.
  They are still rules, so the review page lets you move any prompt to
  another topic before approving. Each prompt is compared on one pair out of
  the six possible, so 150 picks spread across 12 topics leave the smaller
  rows at "too few picks". That's the honest reading at that n, not a bug.

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

Every transcript is graded twice, by two unrelated grader families. There is
no single headline table on purpose — the two graders disagree about who
wins, so quoting one set of numbers without naming its grader would be
reporting an artifact:

| category  | model        | graded by gpt-oss-120b | graded by gemini-3.6-flash | n |
|-----------|--------------|-----------------------:|---------------------------:|--:|
| research  | gpt-oss-120b | 0.50                   | 0.33                       | 2 |
| research  | gpt-oss-20b  | 0.50                   | 0.00                       | 2 |
| documents | gpt-oss-120b | 0.88                   | 0.88                       | 4 |
| documents | gpt-oss-20b  | 0.94                   | 0.81                       | 4 |

Advisory and personal have 6 transcripts each, ready for blind pairwise
voting; **none cast yet** — that step needs a human, not the harness.

What matters here is mostly not the numbers:

- **The previous ranking was a measurement artifact.** The table here used to
  read research 0.58/0.38 and documents 0.88/0.62, apparently a clear win for
  the larger model. Those scores were computed on replies the harness had
  truncated at 250 tokens — 82% of every saved reply ended mid-sentence. Rerun
  untruncated, research ties and documents *reverses*. Nothing about the
  models changed; the instrument did. Treat any result here as provisional
  until you know what the harness was doing to the inputs.
- **Both models hard-failed `research_001` for fabricated citations**, which
  is what zeroes that row. This one is corroborated. Two graders from
  unrelated families (`gpt-oss-120b` and `gemini-3.6-flash`) independently
  reached the same verdict and independently named overlapping invented
  sources. Spot-checking by hand confirms it: the answer cites *"Language
  models can (still) be fooled: the limits of RLHF"* to Gao et al. 2023, but
  Gao et al. 2023 is *Scaling Laws for Reward Model Overoptimization* — real
  authors, real year, invented title and invented finding. Same pattern for
  *"Reward modeling for large language models"* attributed to Bai et al.
  2022. Checking only the author-year passes these; the fabrication is in the
  title.
- **Which model "wins" is decided by which grader you ask.** Every transcript
  was graded twice, by `gpt-oss-120b` and by `gemini-3.6-flash`:

  | category  | model        | gpt-oss grader | gemini grader |
  |-----------|--------------|---------------:|--------------:|
  | documents | gpt-oss-120b | 0.88           | 0.88          |
  | documents | gpt-oss-20b  | **0.94**       | 0.81          |
  | research  | gpt-oss-120b | 0.50           | **0.33**      |
  | research  | gpt-oss-20b  | 0.50           | 0.00          |

  Under the gpt-oss grader, 20b wins `documents` and `research` is a tie.
  Under the Gemini grader, 120b wins both. The ranking is not a property of
  the models here; it is a property of the grader. That is the single most
  important result this harness has produced, and it argues for reporting
  every future number with its grader named.

- **The disagreement rate alone is the wrong statistic.** Per-transcript:

  | category  | disagreement |
  |-----------|-------------:|
  | documents | 1/8  (12%)   |
  | research  | 2/4  (50%)   |
  | overall   | 3/12 (25%)   |

  `documents` sits *below* eval-spec.md's 20% unusable threshold and would
  pass a naive check — yet its single disagreeing row is enough to reverse
  the ranking, because the margin between the models (0.06) is smaller than
  one flipped criterion. A low disagreement rate does not imply a stable
  conclusion. What matters is disagreement *relative to the gap being
  measured*, and with n=4 and n=2 per model that gap is far too small to
  survive any grader noise at all.

- **The two categories are not equally gradeable.** 12% disagreement on
  `documents` versus 50% on `research` is a large, consistent split, and it
  has an obvious explanation: documents criteria are largely structural
  (is it organised, does it hit the required sections) while research
  criteria require judging whether factual claims and citations hold up —
  exactly the judgment a language model is least able to make. Model-graded
  scoring looks viable for the structural category and not for the factual
  one.

  Second-grader output lives in `/scores_gemini` alongside `/scores` rather
  than replacing it, since the disagreement *is* the finding.

- **`hard_fail` is not trustworthy either. This supersedes an earlier claim
  here that it was.** When both graders agreed that `research_001`'s gpt-oss
  answers fabricated citations, that looked like the one reliable signal in
  the set. Adding Gemini models as *subjects* broke it.

  The `gpt-oss-120b` grader hard-failed `gemini-3.5-flash` on `research_001`
  for "attributing findings to fabricated papers (e.g., Gao et al., 2022
  *Scaling Laws for Reward Model Overoptimization*)". That paper is real —
  arXiv 2210.10760 — and the citation is exact. So are *Learning to summarize
  with human feedback* (Stiennon et al. 2020) and *Direct Preference
  Optimization* (Rafailov et al. 2023), also cited in the same answer. Its
  only genuine defects are two title slips: Perez et al. 2022 is
  *Model-Written Evaluations*, and Ziegler et al. 2019 is *from Human
  Preferences*.

  So the grader gave 0.00 to the *better-cited* answer. The gpt-oss answer
  that also scored 0.00 had invented titles wholesale. Identical scores,
  opposite realities.

  The shape of the error matters more than the error. This is not random
  noise that averages out across runs: the grader is wrong about *one
  specific paper in both directions*, correctly catching an invented Gao
  title and then rejecting the genuine one. Error correlated with the claim
  rather than with the sample cannot be fixed by drawing more samples.
  **Model-graded citation checking does not work here, and the `research`
  scores should not be used at all** — not as a ranking, not as a floor, not
  as evidence a model fabricates.

  (Checked against my own knowledge of these papers, which has exactly the
  failure mode this bullet is about. The hand-check is what settles it.)
- **The model grader is not deterministic either.** Re-running it on an
  unchanged transcript once flipped a criterion and moved the score from 1.00
  to 0.75 with no other input changed. Between that and the cross-grader
  disagreement above, a single automated pass is not evidence of anything on
  its own.
- **n is far too small for any of this to be a ranking.** Four transcripts
  per model for `documents`, two for `research`, one run each. The daily
  token cap is what enforces that (see limitations), and it is why no
  variance estimate exists. Every number above should be read as "this
  pipeline runs and produces plausible output", not as a result.

## Grading

Objective sets (`research.yaml`, `documents.yaml`): binary checklist per
`eval-spec.md`, model-graded first pass, then hand-grade a random 15% and
report the disagreement rate. Above ~20% disagreement the automated scores
aren't usable.

Advisory: blind pairwise only, on `held_position`, `specificity`,
`context_retention`, `honesty`, `overall`. Model identities hidden until
the vote is recorded; log the date, since preferences drift.

## License

MIT (see [LICENSE](LICENSE)).
