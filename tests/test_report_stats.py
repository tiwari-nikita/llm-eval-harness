import json

import report_stats
from report_stats import build_report, met_total, model_labels


def _score(d, task, provider, model, crit, hard_fail=False, grader="g1"):
    met = sum(crit.values())
    rec = {
        "task_id": task, "provider": provider, "model": model, "run_index": 0,
        "score": 0.0 if hard_fail else met / len(crit),
        "criteria_met": crit, "hard_fail_triggered": hard_fail,
        "grader_model": grader,
    }
    name = f"{task}__{provider}_{model.replace('/', '_')}__run0.json"
    d.mkdir(exist_ok=True)
    (d / name).write_text(json.dumps(rec), encoding="utf-8")
    return name


def _two_model_set(d, grader="g1", flip=False):
    yes = {"c1": True, "c2": True, "c3": True, "c4": True}
    three = {"c1": True, "c2": True, "c3": True, "c4": False}
    for t in ("documents_001", "documents_002"):
        _score(d, t, "groq", "openai/big", yes, grader=grader)
        _score(d, t, "groq", "openai/small", three if (t == "documents_001" and not flip) else yes,
               grader=grader)


def test_met_total_counts_a_hard_fail_as_nothing_met():
    rec = {"criteria_met": {"c1": True, "c2": True}, "hard_fail_triggered": True}
    assert met_total(rec) == (0, 2)
    rec["hard_fail_triggered"] = False
    assert met_total(rec) == (2, 2)


def test_model_labels_shorten_unless_two_providers_collide():
    recs = [{"provider": "groq", "model": "openai/gpt-oss-20b"},
            {"provider": "google", "model": "gemini-flash"},
            {"provider": "other", "model": "vendor/gemini-flash"}]
    labels = model_labels(recs)
    assert labels[("groq", "openai/gpt-oss-20b")] == "gpt-oss-20b"
    assert labels[("google", "gemini-flash")] == "google/gemini-flash"
    assert labels[("other", "vendor/gemini-flash")] == "other/vendor/gemini-flash"


def test_score_report_shows_means_intervals_and_refuses_a_verdict_on_two_tasks(tmp_path):
    _two_model_set(tmp_path / "s")
    out = build_report(tmp_path / "s", n_boot=500)
    assert "| big | 2 | 1.00 |" in out
    assert "| small | 2 | 0.88 |" in out
    assert "| big vs small | 2 | +0.12 |" in out
    assert "too few tasks for any verdict" in out
    # The lead is one criterion on one task, so a single flip erases it.
    assert out.rstrip().splitlines()[-1].endswith("| 1 |")


def test_a_single_score_has_no_interval(tmp_path):
    _score(tmp_path / "s", "research_001", "groq", "m", {"c1": True, "c2": False})
    out = build_report(tmp_path / "s", n_boot=200)
    assert "n/a (n=1)" in out


def test_agreement_report_counts_disagreements_kappa_and_a_ranking_flip(tmp_path):
    _two_model_set(tmp_path / "a", grader="grader-one")
    _two_model_set(tmp_path / "b", grader="grader-two", flip=True)
    # Second grader also marks big's c4 unmet on documents_002, so small wins.
    rec_path = tmp_path / "b" / "documents_002__groq_openai_big__run0.json"
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    rec["criteria_met"]["c4"] = False
    rec["score"] = 0.75
    rec_path.write_text(json.dumps(rec), encoding="utf-8")

    out = build_report(tmp_path / "a", second=tmp_path / "b", n_boot=200)
    assert "First grader: grader-one. Second grader: grader-two. 4 transcripts scored by both." in out
    assert "| documents | 4 | 2/4 (" in out
    # 16 criterion judgments, 14 agree: 88% raw agreement.
    assert "| 16 | 88% |" in out
    assert "documents: first grader puts big on top, second grader puts small on top" in out


def test_agreement_report_says_so_when_the_ranking_holds(tmp_path):
    _two_model_set(tmp_path / "a")
    _two_model_set(tmp_path / "b")
    out = build_report(tmp_path / "a", second=tmp_path / "b", n_boot=200)
    assert "The top model is the same under both graders in every category." in out


def test_hand_section_reports_the_disagreement_rate_with_an_interval(tmp_path):
    _two_model_set(tmp_path / "s")
    hand = tmp_path / "hand.json"
    hand.write_text(json.dumps([
        {"file": "x.json", "model_score": 1.0, "hand_score": 1.0, "agree": True},
        {"file": "y.json", "model_score": 1.0, "hand_score": 0.5, "agree": False},
    ]), encoding="utf-8")
    out = build_report(tmp_path / "s", hand=hand, n_boot=200)
    assert "Disagreement: 1/2 (" in out


def test_archived_scores_in_subfolders_are_ignored(tmp_path):
    _two_model_set(tmp_path / "s")
    _score(tmp_path / "s" / "_archive", "documents_001", "groq", "openai/old", {"c1": False})
    out = build_report(tmp_path / "s", n_boot=200)
    assert "openai/old" not in out


def test_main_writes_the_report_to_a_file(tmp_path, monkeypatch, capsys):
    _two_model_set(tmp_path / "s")
    out_file = tmp_path / "report.md"
    monkeypatch.setattr("sys.argv", ["report_stats.py", str(tmp_path / "s"), "--boot", "200",
                                     "--out", str(out_file)])
    report_stats.main()
    printed = capsys.readouterr().out
    assert out_file.read_text(encoding="utf-8").strip() == printed.strip()


def test_hand_section_says_so_when_there_are_no_hand_grades_yet(tmp_path):
    _two_model_set(tmp_path / "s")
    out = build_report(tmp_path / "s", hand=tmp_path / "missing.json", n_boot=200)
    assert "No hand grades recorded yet" in out
