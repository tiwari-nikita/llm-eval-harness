import json
import pickle

import pytest

import chat_vote


def _transcript(task_id="advisory_001", model="m-a", turns=2, truncated_at=(),
                system="be brief"):
    return {
        "task_id": task_id,
        "provider": "groq",
        "model": model,
        "max_tokens": 1500,
        "system": system,
        "n_truncated": len(truncated_at),
        "exchanges": [
            {
                "user": f"user turn {i}",
                "assistant": f"reply {i} body",
                "completion_tokens": 100 + i,
                "truncated": i in truncated_at,
            }
            for i in range(1, turns + 1)
        ],
    }


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A throwaway cwd with runs/ so state and preference files land in tmp."""
    (tmp_path / "runs").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(chat_vote, "STATE_DIR", tmp_path / ".vote_state")
    return tmp_path


def _write_pair(project, task_id="advisory_001", **kw):
    for model in ("gpt-oss-120b", "gpt-oss-20b"):
        t = _transcript(task_id=task_id, model=model, **kw)
        path = project / "runs" / f"{task_id}__groq_openai_{model}__run0.json"
        path.write_text(json.dumps(t), encoding="utf-8")


def _show(task_id="advisory_001"):
    chat_vote.cmd_show(_ns(task_id=task_id, task_file=None))


def _ns(**kw):
    import argparse
    return argparse.Namespace(**kw)


def test_show_prints_both_transcripts_without_naming_models(project, capsys):
    _write_pair(project)
    _show()
    out = capsys.readouterr().out
    assert "TRANSCRIPT A" in out and "TRANSCRIPT B" in out
    assert "reply 1 body" in out  # the transcripts really were printed
    # the whole point of blinding: identities must not leak into what is shown
    assert "gpt-oss-120b" not in out
    assert "gpt-oss-20b" not in out


def test_show_hides_the_mapping_in_state_rather_than_stdout(project, capsys):
    _write_pair(project)
    _show()
    state = pickle.loads((project / ".vote_state" / "advisory_001.pkl").read_bytes())
    assert sorted(state.keys()) == ["files", "order"]
    assert sorted(state["order"]) == [0, 1]


def test_show_refuses_a_transcript_that_failed_to_run(project):
    """An error record has no exchanges; voting on it would record a verdict
    about a run that never happened."""
    good = _transcript(model="gpt-oss-20b")
    bad = {"task_id": "advisory_001", "provider": "groq", "model": "gpt-oss-120b",
           "error": "429 Client Error: Too Many Requests"}
    (project / "runs" / "advisory_001__groq_a.json").write_text(json.dumps(bad), encoding="utf-8")
    (project / "runs" / "advisory_001__groq_b.json").write_text(json.dumps(good), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        _show()
    assert "no exchanges" in str(e.value)


def test_show_requires_exactly_two_transcripts(project):
    t = _transcript()
    (project / "runs" / "advisory_001__groq_only.json").write_text(json.dumps(t), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        _show()
    assert "expected exactly 2" in str(e.value)


def test_record_writes_votes_and_reveals_identities(project, capsys):
    _write_pair(project)
    _show()
    capsys.readouterr()
    chat_vote.cmd_record(_ns(task_id="advisory_001", task_file=None,
                             votes="a,b,tie,a,b", category="advisory",
                             preference_dir="preference"))
    out = capsys.readouterr().out
    assert "Revealed:" in out

    written = list((project / "preference").glob("advisory_001__*.json"))
    assert len(written) == 1
    rec = json.loads(written[0].read_text(encoding="utf-8"))
    assert rec["votes"] == {
        "held_position": "a", "specificity": "b", "context_retention": "tie",
        "honesty": "a", "overall": "b",
    }
    assert {rec["model_a"], rec["model_b"]} == {
        "groq/gpt-oss-120b", "groq/gpt-oss-20b"}
    assert rec["elicited_via"] == "chat"


def test_record_carries_truncation_into_the_vote_record(project):
    """gpt-oss-120b overran a brevity instruction 20b followed and was cut off.
    That counts against it -- but only if the record says it happened, or a
    later reader will blame the harness truncation bug instead."""
    _write_pair(project, truncated_at=(2,))
    _show()
    chat_vote.cmd_record(_ns(task_id="advisory_001", task_file=None,
                             votes="a,a,a,a,a", category="advisory",
                             preference_dir="preference"))
    rec = json.loads(next((project / "preference").glob("*.json")).read_text(encoding="utf-8"))
    for side in ("transcript_a_meta", "transcript_b_meta"):
        assert rec[side]["truncated_turns"] == [2]
        assert rec[side]["max_tokens"] == 1500
        assert rec[side]["system"] == "be brief"
        assert rec[side]["completion_tokens"] == [101, 102]


def test_record_rejects_a_vote_count_that_misses_a_dimension(project):
    _write_pair(project)
    _show()
    with pytest.raises(SystemExit) as e:
        chat_vote.cmd_record(_ns(task_id="advisory_001", task_file=None,
                                 votes="a,b", category="advisory",
                                 preference_dir="preference"))
    assert "expected 5 votes" in str(e.value)


def test_record_rejects_an_unrecognised_vote_value(project):
    _write_pair(project)
    _show()
    with pytest.raises(SystemExit) as e:
        chat_vote.cmd_record(_ns(task_id="advisory_001", task_file=None,
                                 votes="a,b,maybe,a,a", category="advisory",
                                 preference_dir="preference"))
    assert "invalid vote" in str(e.value)


def test_record_without_a_prior_show_refuses_rather_than_guessing(project):
    _write_pair(project)
    with pytest.raises(SystemExit) as e:
        chat_vote.cmd_record(_ns(task_id="advisory_001", task_file=None,
                                 votes="a,a,a,a,a", category="advisory",
                                 preference_dir="preference"))
    assert "no shown-state" in str(e.value)


def test_recording_consumes_the_state_so_a_vote_cannot_be_replayed(project):
    _write_pair(project)
    _show()
    args = _ns(task_id="advisory_001", task_file=None, votes="a,a,a,a,a",
               category="advisory", preference_dir="preference")
    chat_vote.cmd_record(args)
    assert not (project / ".vote_state" / "advisory_001.pkl").exists()
    with pytest.raises(SystemExit):
        chat_vote.cmd_record(args)


def test_personal_votes_are_prefixed_so_gitignore_keeps_them_local(project):
    _write_pair(project, task_id="personal_001")
    _show(task_id="personal_001")
    chat_vote.cmd_record(_ns(task_id="personal_001", task_file=None,
                             votes="a,a,a,a,a", category="personal",
                             preference_dir="preference"))
    names = [p.name for p in (project / "preference").glob("*.json")]
    assert len(names) == 1
    assert names[0].startswith("personal_personal_001__")


def test_dimensions_come_from_the_task_file_when_it_has_them(project):
    _write_pair(project)
    task_file = project / "advisory.yaml"
    task_file.write_text("dimensions:\n  - warmth\n  - rigour\n", encoding="utf-8")
    assert chat_vote.load_dimensions(str(task_file)) == ["warmth", "rigour"]


def test_dimensions_fall_back_to_the_standard_set(project):
    assert chat_vote.load_dimensions(None) == chat_vote.DEFAULT_DIMENSIONS
    assert chat_vote.load_dimensions("does_not_exist.yaml") == chat_vote.DEFAULT_DIMENSIONS
