import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

import score_objective
from score_objective import cmd_grade

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


# --- the runner boundary ----------------------------------------------------
# cmd_grade had no coverage, so when call_model's return type changed from a
# string to a dict nothing failed until a live grading run crashed with
# "'dict' object has no attribute 'find'". These tests mock call_model with
# its real shape, so the contract is pinned from both sides.

TASK_YAML = [{
    "id": "research_001",
    "category": "research",
    "prompt": "q",
    "criteria": [
        {"id": "c1", "text": "cites a source", "type": "content"},
        {"id": "c2", "text": "states a limitation", "type": "reasoning"},
    ],
    "hard_fail": ["invents a citation"],
}]

VERDICT = {
    "criteria_met": {"c1": True, "c2": True},
    "hard_fail_triggered": False,
    "hard_fail_reason": None,
}


def _grade_setup(tmp_path, reply):
    task_file = tmp_path / "task.yaml"
    task_file.write_text(yaml.safe_dump(TASK_YAML), encoding="utf-8")

    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "research_001__groq_m__run0.json").write_text(json.dumps({
        "task_id": "research_001", "provider": "groq", "model": "m",
        "run_index": 0, "exchanges": [{"user": "q", "assistant": "an answer"}],
    }), encoding="utf-8")

    scores = tmp_path / "scores"
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump(
        {"groq": {"base_url": "http://x", "api_key_env": "GROQ_API_KEY",
                  "models": ["m"]}}), encoding="utf-8")

    args = MagicMock(task=str(task_file), runs_dir=str(runs),
                     scores_dir=str(scores), providers_file=str(providers),
                     grader_provider="groq", grader_model="m",
                     grader_max_tokens=3000, skip_existing=False)
    return args, scores


def test_cmd_grade_reads_content_out_of_the_call_model_dict(tmp_path, monkeypatch):
    """Regression: call_model returns a dict, not a string."""
    monkeypatch.setenv("GROQ_API_KEY", "k")
    args, scores = _grade_setup(tmp_path, None)
    reply = {"content": json.dumps(VERDICT), "finish_reason": "stop",
             "completion_tokens": 20, "total_tokens": 100}
    with patch.object(score_objective, "call_model", return_value=reply):
        cmd_grade(args)
    written = list(Path(scores).glob("*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text(encoding="utf-8"))["score"] == 1.0


def test_cmd_grade_skips_a_verdict_truncated_by_the_token_cap(tmp_path, monkeypatch,
                                                              capsys):
    """A verdict cut off mid-JSON must not be scored -- a partial criteria
    dict grades the answer lower than it deserves."""
    monkeypatch.setenv("GROQ_API_KEY", "k")
    args, scores = _grade_setup(tmp_path, None)
    reply = {"content": '{"criteria_met": {"c1": tru',
             "finish_reason": "length", "completion_tokens": 600,
             "total_tokens": 900}
    with patch.object(score_objective, "call_model", return_value=reply):
        cmd_grade(args)
    assert not list(Path(scores).glob("*.json"))
    assert "hit the token cap" in capsys.readouterr().err


def test_cmd_grade_skip_existing_leaves_a_recorded_score_alone(tmp_path, monkeypatch):
    """The grader is non-deterministic, so an unnecessary re-grade can rewrite
    a recorded score with a different one and destroy the comparison it was
    run to produce."""
    monkeypatch.setenv("GROQ_API_KEY", "k")
    args, scores = _grade_setup(tmp_path, None)
    good = {"content": json.dumps(VERDICT), "finish_reason": "stop",
            "completion_tokens": 20, "total_tokens": 100}
    with patch.object(score_objective, "call_model", return_value=good):
        cmd_grade(args)
    first = json.loads(list(Path(scores).glob("*.json"))[0].read_text(encoding="utf-8"))
    assert first["score"] == 1.0

    args.skip_existing = True
    flipped = {"content": json.dumps({"criteria_met": {"c1": False, "c2": False},
                                      "hard_fail_triggered": False,
                                      "hard_fail_reason": None}),
               "finish_reason": "stop", "completion_tokens": 20, "total_tokens": 100}
    with patch.object(score_objective, "call_model", return_value=flipped) as m:
        cmd_grade(args)
    assert m.call_count == 0
    after = json.loads(list(Path(scores).glob("*.json"))[0].read_text(encoding="utf-8"))
    assert after["score"] == 1.0


# --- hand-sample -----------------------------------------------------------

import argparse


def _scored(d, name, score):
    d.mkdir(exist_ok=True)
    (d / name).write_text(json.dumps({"score": score, "source_transcript": name,
                                      "grader_model": "g"}), encoding="utf-8")


def _hand_args(scores_dir, out, fraction=1.0, seed=0):
    return argparse.Namespace(scores_dir=str(scores_dir), fraction=fraction, out=str(out), seed=seed)


def test_hand_sample_saves_every_grade_and_reports_an_interval(tmp_path, monkeypatch, capsys):
    s = tmp_path / "scores"
    _scored(s, "a.json", 1.0)
    _scored(s, "b.json", 0.75)
    answers = iter(["agree", "0.5"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    out = tmp_path / "hand" / "hand_grades.json"
    score_objective.cmd_hand_sample(_hand_args(s, out))

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert len(saved) == 2
    assert sum(r["agree"] for r in saved) == 1
    agreed = [r for r in saved if r["agree"]][0]
    assert agreed["hand_score"] == agreed["model_score"]
    printed = capsys.readouterr().out
    assert "Disagreement rate: 1/2 = 50% (95% interval" in printed
    assert "Saved 2 hand grades" in printed


def test_hand_sample_regrading_a_file_replaces_its_old_entry(tmp_path, monkeypatch):
    s = tmp_path / "scores"
    _scored(s, "a.json", 1.0)
    out = tmp_path / "hand_grades.json"
    monkeypatch.setattr("builtins.input", lambda _prompt: "0.0")
    score_objective.cmd_hand_sample(_hand_args(s, out))
    monkeypatch.setattr("builtins.input", lambda _prompt: "agree")
    score_objective.cmd_hand_sample(_hand_args(s, out))
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert len(saved) == 1
    assert saved[0]["agree"] is True


def test_hand_sample_non_number_counts_as_disagreement(tmp_path, monkeypatch):
    s = tmp_path / "scores"
    _scored(s, "a.json", 1.0)
    out = tmp_path / "hand_grades.json"
    monkeypatch.setattr("builtins.input", lambda _prompt: "dunno")
    score_objective.cmd_hand_sample(_hand_args(s, out))
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved[0]["hand_score"] is None and saved[0]["agree"] is False


def test_hand_sample_with_a_seed_picks_the_same_files(tmp_path, monkeypatch):
    s = tmp_path / "scores"
    for i in range(10):
        _scored(s, f"f{i}.json", 1.0)
    monkeypatch.setattr("builtins.input", lambda _prompt: "agree")
    picks = []
    for run in range(2):
        out = tmp_path / f"h{run}.json"
        score_objective.cmd_hand_sample(_hand_args(s, out, fraction=0.3, seed=7))
        picks.append(sorted(r["file"] for r in json.loads(out.read_text(encoding="utf-8"))))
    assert picks[0] == picks[1]
    assert len(picks[0]) == 3
