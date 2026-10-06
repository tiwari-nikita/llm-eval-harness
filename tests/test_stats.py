import pytest

from stats import (bootstrap_mean_ci, cohen_kappa, flips_to_erase_lead,
                   paired_difference, sign_test, wilson_interval,
                   MIN_TASKS_FOR_SIGN_TEST)


# --- Wilson ---------------------------------------------------------------

def test_wilson_matches_the_textbook_value_for_3_of_12():
    lo, hi = wilson_interval(3, 12)
    assert lo == pytest.approx(0.0889, abs=1e-4)
    assert hi == pytest.approx(0.5323, abs=1e-4)


def test_wilson_at_zero_successes_starts_at_zero_and_is_not_degenerate():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.2775, abs=1e-4)


def test_wilson_is_symmetric_between_k_and_n_minus_k():
    lo, hi = wilson_interval(2, 8)
    lo2, hi2 = wilson_interval(6, 8)
    assert lo == pytest.approx(1 - hi2)
    assert hi == pytest.approx(1 - lo2)


@pytest.mark.parametrize("k,n", [(1, 0), (-1, 5), (6, 5)])
def test_wilson_rejects_impossible_counts(k, n):
    with pytest.raises(ValueError):
        wilson_interval(k, n)


# --- bootstrap ------------------------------------------------------------

def test_bootstrap_of_one_value_has_no_interval():
    assert bootstrap_mean_ci([0.75]) == (0.75, None, None)


def test_bootstrap_of_identical_values_collapses_to_the_value():
    assert bootstrap_mean_ci([0.5, 0.5, 0.5]) == (0.5, 0.5, 0.5)


def test_bootstrap_is_reproducible_for_a_seed_and_brackets_the_mean():
    vals = [1.0, 0.75, 0.75, 0.5]
    first = bootstrap_mean_ci(vals, seed=3)
    assert first == bootstrap_mean_ci(vals, seed=3)
    m, lo, hi = first
    assert m == pytest.approx(0.75)
    assert 0.5 <= lo <= m <= hi <= 1.0


def test_bootstrap_rejects_empty_input():
    with pytest.raises(ValueError):
        bootstrap_mean_ci([])


# --- sign test ------------------------------------------------------------

def test_sign_test_four_clean_wins_is_still_not_significant():
    assert sign_test([0.1, 0.2, 0.1, 0.3]) == (4, 0, 0.125)


def test_sign_test_needs_six_clean_wins_to_get_below_five_percent():
    wins, losses, p = sign_test([0.1] * MIN_TASKS_FOR_SIGN_TEST)
    assert (wins, losses) == (6, 0)
    assert p == pytest.approx(2 / 2 ** 6)
    assert p < 0.05
    assert sign_test([0.1] * (MIN_TASKS_FOR_SIGN_TEST - 1))[2] > 0.05


def test_sign_test_drops_ties():
    assert sign_test([0.0, 0.0, 0.25]) == (1, 0, 1.0)
    assert sign_test([0.0, 0.0]) == (0, 0, 1.0)


def test_sign_test_even_split_is_p_one():
    assert sign_test([0.1, -0.1, 0.2, -0.2])[2] == 1.0


# --- paired comparison ----------------------------------------------------

def test_paired_difference_will_not_call_a_winner_on_four_tasks():
    a = {"t1": 1.0, "t2": 1.0, "t3": 0.75, "t4": 1.0}
    b = {"t1": 0.75, "t2": 0.75, "t3": 0.5, "t4": 0.75}
    r = paired_difference(a, b)
    assert r["diff"] == pytest.approx(0.25)
    assert (r["wins"], r["losses"], r["ties"]) == (4, 0, 0)
    assert r["verdict"] == "not distinguishable"


def test_paired_difference_calls_a_clean_sweep_of_six():
    a = {f"t{i}": 1.0 for i in range(6)}
    b = {f"t{i}": 0.5 for i in range(6)}
    assert paired_difference(a, b)["verdict"] == "A ahead"
    assert paired_difference(b, a)["verdict"] == "B ahead"


def test_paired_difference_only_counts_shared_tasks():
    a = {"t1": 1.0, "t2": 0.0, "only_a": 1.0}
    b = {"t1": 0.5, "t2": 0.5, "only_b": 0.0}
    r = paired_difference(a, b)
    assert r["n"] == 2
    assert r["diff"] == pytest.approx(0.0)


def test_paired_difference_with_nothing_shared_raises():
    with pytest.raises(ValueError):
        paired_difference({"t1": 1.0}, {"t2": 1.0})


# --- kappa ----------------------------------------------------------------

def _pairs(tt, tf, ft, ff):
    return [(True, True)] * tt + [(True, False)] * tf + [(False, True)] * ft + [(False, False)] * ff


def test_kappa_matches_the_textbook_example():
    # 50 items: 20 both yes, 5 yes/no, 10 no/yes, 15 both no -> kappa 0.4
    r = cohen_kappa(_pairs(20, 5, 10, 15))
    assert r["n"] == 50
    assert r["agreement"] == pytest.approx(0.7)
    assert r["expected"] == pytest.approx(0.5)
    assert r["kappa"] == pytest.approx(0.4)


def test_kappa_is_one_for_perfect_agreement_with_both_labels_used():
    assert cohen_kappa(_pairs(3, 0, 0, 2))["kappa"] == pytest.approx(1.0)


def test_kappa_is_negative_when_raters_systematically_disagree():
    assert cohen_kappa(_pairs(0, 3, 3, 0))["kappa"] == pytest.approx(-1.0)


def test_high_raw_agreement_can_be_chance_level_kappa():
    # Both graders say "met" to 90% of criteria: 82% raw agreement, kappa 0.
    r = cohen_kappa(_pairs(81, 9, 9, 1))
    assert r["agreement"] == pytest.approx(0.82)
    assert r["kappa"] == pytest.approx(0.0, abs=1e-9)


def test_kappa_is_undefined_when_both_raters_never_vary():
    r = cohen_kappa(_pairs(6, 0, 0, 0))
    assert r["agreement"] == 1.0
    assert r["kappa"] is None


def test_kappa_rejects_empty_input():
    with pytest.raises(ValueError):
        cohen_kappa([])


# --- flips to erase a lead ------------------------------------------------

def test_one_flip_erases_a_one_criterion_lead():
    leader = {"t1": (4, 4), "t2": (4, 4)}
    trailer = {"t1": (4, 4), "t2": (3, 4)}
    assert flips_to_erase_lead(leader, trailer) == 1


def test_no_lead_needs_no_flips():
    same = {"t1": (3, 4)}
    assert flips_to_erase_lead(same, same) == 0
    assert flips_to_erase_lead({"t1": (2, 4)}, {"t1": (3, 4)}) == 0


def test_cheapest_flips_come_from_tasks_with_fewest_criteria():
    # Gap = (1/2 + 0) / 2 tasks = 1/4. A flip on the 2-criterion task moves the
    # mean by 1/4 by itself; flips on the 5-criterion task only move it 1/10.
    leader = {"small": (2, 2), "big": (5, 5)}
    trailer = {"small": (1, 2), "big": (5, 5)}
    assert flips_to_erase_lead(leader, trailer) == 1


def test_bigger_leads_need_more_flips():
    leader = {"t1": (4, 4), "t2": (4, 4)}
    trailer = {"t1": (2, 4), "t2": (2, 4)}
    # Gap 1/2; each flip is worth 1/8 of the mean, so 4 flips.
    assert flips_to_erase_lead(leader, trailer) == 4


def test_flips_with_nothing_shared_raises():
    with pytest.raises(ValueError):
        flips_to_erase_lead({"a": (1, 1)}, {"b": (1, 1)})
