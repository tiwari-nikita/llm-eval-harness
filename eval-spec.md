# Eval Spec (v2)

The contract for how models get scored. Written before any outputs are seen.

Changed from v1: multi-turn advisory is now the main measurement, not a
side category. The objective sets stay as a sanity check. Personal
conversations are local-only and never committed.

## Repo layout

```
/tasks              task files, one per category
/prompts            YOUR extracted prompts — gitignored
/runs               raw transcripts, one JSON per (task, model, run)
/scores             graded objective results
/preference         blind pairwise votes
/router             selection logic reading /scores and /preference
```

`.gitignore` must contain `/prompts` and `/runs`. Ship the schema and a
sample file; anyone cloning writes their own.

## Categories and how each is measured

| Category  | Method            | Committed to repo |
|-----------|-------------------|-------------------|
| advisory  | blind pairwise    | yes               |
| research  | binary checklist  | yes               |
| documents | binary checklist  | yes               |
| personal  | blind pairwise    | no, local only    |

Agentic and app-integration are out of scope. They test the harness, not
the model.

Weighting in the router: advisory dominates, because it is most of the
real usage. The checklist categories exist to catch a model that is
pleasant and wrong.

## Objective task schema

```yaml
- id: research_001
  category: research
  prompt: |
    <exact text sent to the model, no edits at run time>
  criteria:                  # each binary: met or not met
    - id: c1
      text: <a single checkable thing the answer must contain or do>
      type: content | reasoning | format
  hard_fail:
    - states a specific fact, figure, citation or quote that is fabricated
  notes: <why this task is in the set>
```

Binary, not 1-5. Scales are not reproducible across grading sessions and
you will drift.

Score = met / total, unless a hard_fail triggers, in which case zero. A
fluent wrong answer is worse than a refusal and the scoring has to say so.

## Multi-turn schema

```yaml
- id: advisory_001
  title: <what this script is testing>
  turns:
    - <turn 1, verbatim, identical for every model>
    - <turn 2 — must be reply-agnostic>
  probes:
    - <what to watch for when reading the transcript>
```

Turns are fixed. Branching would mean comparing different conversations
rather than different models, and no difference could be attributed.
Write turns that work regardless of the reply: "push back on that" is
fine, "yes, use the second option" is not.

The artificiality of fixed turns is a real limitation and goes in the
README rather than being hidden.

## Blind pairwise protocol

1. Two transcripts, model identities hidden.
2. Randomise which is shown as A.
3. Rank on each dimension, then overall.
4. Reveal only after the vote is written.
5. Log the date. Preferences drift and you want to see it.

Dimensions: held_position, specificity, context_retention, honesty,
overall.

Pairwise beats star ratings. People rate inconsistently on a scale and
choose reliably between two things.

This is N-of-1 and does not generalise. The README says so. A router that
learns one person's preferences is the point.

## Grading the objective sets

Model-graded first pass, then hand-grade a random 15% and report the
disagreement rate in the README.

Known grader biases to look for: preference for longer answers, and
preference for outputs from the grader's own model family.

Above roughly 20% disagreement the automated scores are not usable and
the rubric needs tightening.

## Building the prompt corpus

Export your history from each tool's settings, then run
`extract_prompts.py`. Read the bucket distribution before writing any
tasks: it tells you what you actually use models for, which is probably
not what you think.

Turn real prompts into tasks. Do not paste them in raw, since most single
turns lack the context that made them answerable.

## Router

Reads scores and preference records, returns a ranked model list for a
given category, filtered by what is currently free and within quota.

Provider and free-tier data comes from an existing maintained catalogue,
not from a list you keep by hand. That fight is already lost and it is
not the interesting part.

Recommendations are about fit and quota, never about paid-tier quality.
Free and paid tiers are often different models, and inferring one from
the other is not supportable.
