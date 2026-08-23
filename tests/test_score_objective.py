import json

import pytest

from score_objective import extract_json, compute_score


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_surrounding_prose():
    text = 'Sure, here is the verdict:\n{"a": 1, "b": [1,2]}\nHope that helps!'
    assert extract_json(text) == {"a": 1, "b": [1, 2]}


def test_extract_json_no_braces_raises():
    with pytest.raises(ValueError):
        extract_json("no json here at all")


TASK = {
    "id": "t1",
    "prompt": "irrelevant",
    "criteria": [{"id": "c1", "text": "x"}, {"id": "c2", "text": "y"}, {"id": "c3", "text": "z"}],
    "hard_fail": ["fabricates a citation"],
}


def test_compute_score_all_met():
    verdict = {"criteria_met": {"c1": True, "c2": True, "c3": True}, "hard_fail_triggered": False}
    score, reason = compute_score(TASK, verdict)
    assert score == 1.0
    assert reason is None


def test_compute_score_partial():
    verdict = {"criteria_met": {"c1": True, "c2": False, "c3": True}, "hard_fail_triggered": False}
    score, reason = compute_score(TASK, verdict)
    assert score == pytest.approx(2 / 3)


def test_compute_score_hard_fail_zeroes_regardless_of_criteria():
    verdict = {
        "criteria_met": {"c1": True, "c2": True, "c3": True},
        "hard_fail_triggered": True,
        "hard_fail_reason": "invented a paper",
    }
    score, reason = compute_score(TASK, verdict)
    assert score == 0.0
    assert reason == "invented a paper"


def test_compute_score_missing_criterion_counts_as_not_met():
    verdict = {"criteria_met": {"c1": True}, "hard_fail_triggered": False}
    score, _ = compute_score(TASK, verdict)
    assert score == pytest.approx(1 / 3)
