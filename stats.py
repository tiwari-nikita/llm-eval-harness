"""
Small-sample statistics for eval results: intervals, paired model comparisons,
and grader agreement.

Standard library only, and deterministic: every bootstrap takes a seed, so a
report rebuilt from the same score files prints the same numbers.

Why these and not a t-test: the sets here are tiny (2-4 tasks per model per
category) and the scores are bounded fractions, so anything that leans on a
normal approximation is worse than useless. Wilson intervals behave at small n
and near 0 and 1; the sign test is exact; the bootstrap is descriptive, and is
labelled as too narrow when n is small rather than dressed up as more.
"""

import math
import random
from fractions import Fraction
from statistics import mean

Z95 = 1.959963984540054

# An exact two-sided sign test can't get below p = 2 / 2**n, so with fewer than
# this many non-tied tasks no result could reach p < 0.05, however lopsided.
MIN_TASKS_FOR_SIGN_TEST = 6


def wilson_interval(k, n, z=Z95):
    """95% Wilson score interval for k successes out of n, as (lo, hi)."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must be between 0 and n")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def _percentile_bounds(sorted_vals, level):
    tail = (1 - level) / 2
    last = len(sorted_vals) - 1
    return (sorted_vals[int(math.floor(tail * last))],
            sorted_vals[int(math.ceil((1 - tail) * last))])


def bootstrap_mean_ci(values, level=0.95, n_boot=10_000, seed=0):
    """Percentile-bootstrap interval for the mean, as (mean, lo, hi).

    With a single value there is nothing to resample, so lo and hi are None:
    report that as "n/a", not as a zero-width interval. With a handful of values
    the percentile bootstrap is too narrow (it can only reshuffle what it has
    seen), so treat it as the least uncertainty there is, not the most.
    """
    values = list(values)
    if not values:
        raise ValueError("no values")
    m = mean(values)
    if len(values) == 1:
        return m, None, None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(mean(rng.choices(values, k=n)) for _ in range(n_boot))
    lo, hi = _percentile_bounds(means, level)
    return m, lo, hi


def sign_test(diffs):
    """Exact two-sided sign test on paired differences. Zero differences (ties)
    are dropped. Returns (wins, losses, p), where wins counts positive diffs."""
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    n = wins + losses
    if n == 0:
        return 0, 0, 1.0
    k = min(wins, losses)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return wins, losses, min(1.0, p)


def paired_difference(a, b, level=0.95, n_boot=10_000, seed=0, alpha=0.05):
    """Compare model A with model B on the tasks both were scored on.

    a and b map task id -> score. The pair for each task stays together when
    resampling, because a hard task drags both models down and that shared
    difficulty isn't evidence for either.

    Returns a dict: n (shared tasks), diff (mean of a - b), lo/hi (bootstrap
    interval on diff, descriptive), wins/losses/ties (per task), p (exact
    sign test), and verdict: "A ahead", "B ahead", or "not distinguishable".
    """
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError("no tasks in common")
    diffs = [a[t] - b[t] for t in shared]
    diff, lo, hi = bootstrap_mean_ci(diffs, level, n_boot, seed)
    wins, losses, p = sign_test(diffs)
    if p < alpha and wins > losses:
        verdict = "A ahead"
    elif p < alpha and losses > wins:
        verdict = "B ahead"
    else:
        verdict = "not distinguishable"
    return {"n": len(shared), "diff": diff, "lo": lo, "hi": hi,
            "wins": wins, "losses": losses, "ties": len(shared) - wins - losses,
            "p": p, "verdict": verdict}


def cohen_kappa(pairs):
    """Cohen's kappa for two raters' binary labels.

    pairs is a list of (label_a, label_b). Raw agreement flatters a rubric
    where almost everything is "met": two graders who both say yes to 90% of
    criteria agree 82% of the time by chance alone. Kappa is agreement beyond
    that chance level: 1 is perfect, 0 is chance, below 0 is worse than chance.

    Returns a dict: n, agreement (observed), expected (chance), kappa. kappa is
    None when it is undefined, i.e. both raters gave one label to everything,
    so chance agreement is already 100%.
    """
    pairs = [(bool(x), bool(y)) for x, y in pairs]
    n = len(pairs)
    if n == 0:
        raise ValueError("no pairs")
    observed = sum(x == y for x, y in pairs) / n
    pa = sum(x for x, _ in pairs) / n
    pb = sum(y for _, y in pairs) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    kappa = None if math.isclose(expected, 1.0) else (observed - expected) / (1 - expected)
    return {"n": n, "agreement": observed, "expected": expected, "kappa": kappa}


def flips_to_erase_lead(leader, trailer):
    """The fewest single-criterion flips that would erase the leader's lead
    (tie it or reverse it), counting only flips that are possible.

    leader and trailer map task id -> (criteria met, criteria total). Only
    shared tasks count. A flip either turns one of the leader's met criteria
    into unmet, or one of the trailer's unmet criteria into met; on a task with
    c criteria that moves a model's mean score by 1 / (c * n_tasks). Hard fails
    are left out on purpose: one of those moves a score all the way to 0, which
    would make every lead look fragile.

    Returns 0 when there is no lead. (Flipping every available criterion always
    closes a gap, so the loop below always returns.)
    """
    shared = sorted(set(leader) & set(trailer))
    if not shared:
        raise ValueError("no tasks in common")
    n = len(shared)
    gap = sum(Fraction(leader[t][0], leader[t][1]) - Fraction(trailer[t][0], trailer[t][1])
              for t in shared) / n
    if gap <= 0:
        return 0
    steps = []
    for t in shared:
        met, total = leader[t]
        steps += [Fraction(1, total * n)] * met
        met, total = trailer[t]
        steps += [Fraction(1, total * n)] * (total - met)
    closed = Fraction(0)
    for count, step in enumerate(sorted(steps, reverse=True), start=1):
        closed += step
        if closed >= gap:
            return count
    raise AssertionError("unreachable: flipping every criterion closes any gap")
