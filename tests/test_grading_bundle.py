import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import make_grading_bundle as mgb
import ingest_votes as iv


ROOT = Path(__file__).resolve().parent.parent


def _transcript(task_id, provider, model, n_turns=2, truncated=()):
    return {
        "task_id": task_id, "provider": provider, "model": model, "run_index": 0,
        "max_tokens": 1500,
        # Neutral body text on purpose: a real transcript does not name the
        # model that produced it, so echoing the id here would fake a leak the
        # blinding test is meant to detect.
        "exchanges": [
            {"user": f"turn {i+1}", "assistant": f"answer {i+1} from {provider[0]}",
             "truncated": (i + 1) in truncated}
            for i in range(n_turns)
        ],
    }


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Point STATE_DIR at a temp dir.

    Without this, every test that builds a bundle drops a real mapping file
    into the project's .vote_state/, mixed in with live ones and impossible to
    tell apart afterwards.
    """
    d = tmp_path / "vote_state"
    d.mkdir()
    monkeypatch.setattr(mgb, "STATE_DIR", d)
    monkeypatch.setattr(iv, "STATE_DIR", d)
    return d


@pytest.fixture
def runs(tmp_path):
    d = tmp_path / "runs"
    d.mkdir()
    for task in ("advisory_001", "advisory_002"):
        for prov, model in (("groq", "openai/gpt-oss-120b"), ("google", "gemini-3.5-flash")):
            slug = model.replace("/", "_")
            (d / f"{task}__{prov}_{slug}__run0.json").write_text(
                json.dumps(_transcript(task, prov, model, truncated=(2,) if prov == "groq" else ())),
                encoding="utf-8")
    return d


def _args(runs_dir, tmp_path, **kw):
    class A:
        pass
    a = A()
    a.category = kw.get("category", "advisory")
    a.runs_dir = str(runs_dir)
    a.task_file = kw.get("task_file")
    a.out = kw.get("out", str(tmp_path / "grade.html"))
    a.include_personal = kw.get("include_personal", False)
    return a


def test_pairs_are_found_per_task(runs):
    pairs = mgb.find_pairs(str(runs), "advisory")
    assert [p[0] for p in pairs] == ["advisory_001", "advisory_002"]


def test_a_task_without_exactly_two_healthy_transcripts_is_skipped(runs, capsys):
    (runs / "advisory_001__groq_openai_gpt-oss-120b__run0.json").unlink()
    pairs = mgb.find_pairs(str(runs), "advisory")
    assert [p[0] for p in pairs] == ["advisory_002"]
    assert "need exactly 2" in capsys.readouterr().err


def test_error_records_are_never_offered_for_grading(runs, capsys):
    """An error record is valid JSON with no exchanges. Rendering it would show
    a blank transcript and collect a verdict on a run that never happened."""
    p = runs / "advisory_001__groq_openai_gpt-oss-120b__run0.json"
    p.write_text(json.dumps({"task_id": "advisory_001", "provider": "groq",
                             "model": "m", "error": "429"}), encoding="utf-8")
    pairs = mgb.find_pairs(str(runs), "advisory")
    assert [t for t, _ in pairs] == ["advisory_002"]
    assert "error record" in capsys.readouterr().err


def test_generated_page_contains_no_model_identities(runs, tmp_path, monkeypatch):
    """Blinding is enforced by construction: the page is sent to other people,
    the mapping is not."""
    monkeypatch.chdir(ROOT)
    args = _args(runs, tmp_path)
    mgb.build(args)
    html = Path(args.out).read_text(encoding="utf-8")
    payload = html.split("const BUNDLE = ")[1].split(";\n")[0]
    low = payload.lower()
    for leak in ("gpt-oss", "gemini", "groq", "openai"):
        assert leak not in low, f"{leak!r} reached the page a grader receives"

    # Structural, not just textual: no identity-bearing key should exist at all,
    # so a future field cannot reintroduce the leak past a substring check.
    bundle = json.loads(payload)
    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from keys(v)
    assert not {"model", "provider", "file"} & set(keys(bundle))


def test_mapping_file_records_both_identities(runs, tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    args = _args(runs, tmp_path)
    mgb.build(args)
    html = Path(args.out).read_text(encoding="utf-8")
    bundle_id = json.loads(html.split("const BUNDLE = ")[1].split(";\n")[0])["bundle_id"]
    state = json.loads((mgb.STATE_DIR / f"bundle_{bundle_id}.json").read_text(encoding="utf-8"))
    models = {s["model"] for m in state["mapping"].values() for s in m.values()}
    assert models == {"openai/gpt-oss-120b", "gemini-3.5-flash"}


def test_truncated_turns_are_surfaced_to_the_grader(runs, tmp_path, monkeypatch):
    """A reply cut off by the cap looks like a model trailing away. The grader
    has to be told which it was."""
    monkeypatch.chdir(ROOT)
    args = _args(runs, tmp_path)
    mgb.build(args)
    html = Path(args.out).read_text(encoding="utf-8")
    bundle = json.loads(html.split("const BUNDLE = ")[1].split(";\n")[0])
    flags = [t["a_truncated"] + t["b_truncated"] for t in bundle["tasks"]]
    assert all(f == [2] for f in flags)


def test_personal_category_refuses_without_the_explicit_flag():
    """These transcripts are real private chat history and the output file is
    meant to be handed to other people."""
    r = subprocess.run([sys.executable, "make_grading_bundle.py",
                        "--category", "personal"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0
    assert "include-personal" in (r.stdout + r.stderr)


def test_script_tag_cannot_be_closed_early_by_transcript_text(runs, tmp_path, monkeypatch):
    """A transcript containing </script> would end the block and break the
    page, or worse, inject markup."""
    monkeypatch.chdir(ROOT)
    p = runs / "advisory_001__groq_openai_gpt-oss-120b__run0.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["exchanges"][0]["assistant"] = "use </script><b>x</b> carefully"
    p.write_text(json.dumps(d), encoding="utf-8")
    args = _args(runs, tmp_path)
    mgb.build(args)
    html = Path(args.out).read_text(encoding="utf-8")
    body = html.split("const BUNDLE = ")[1].split(";\n")[0]
    assert "</script>" not in body
    assert "<\\/script>" in body
    bid = json.loads(body)["bundle_id"]


# --- ingest -----------------------------------------------------------------

@pytest.fixture
def bundle_and_votes(runs, tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    args = _args(runs, tmp_path)
    mgb.build(args)
    html = Path(args.out).read_text(encoding="utf-8")
    bundle = json.loads(html.split("const BUNDLE = ")[1].split(";\n")[0])
    votes_path = tmp_path / "votes.json"
    votes_path.write_text(json.dumps({
        "bundle_id": bundle["bundle_id"], "category": "advisory", "grader": "alice",
        "submitted": "2026-08-26T12:00:00Z",
        "votes": {t["task_id"]: {d["name"]: "a" for d in bundle["dimensions"]}
                  for t in bundle["tasks"]},
        "notes": {bundle["tasks"][0]["task_id"]: "because"},
    }), encoding="utf-8")
    return bundle, votes_path, tmp_path


def test_ingest_resolves_sides_back_to_models(bundle_and_votes):
    bundle, votes_path, tmp_path = bundle_and_votes
    pref = tmp_path / "pref"
    written = iv.ingest_one(str(votes_path), str(pref))
    assert len(written) == len(bundle["tasks"])
    rec = json.loads(next(pref.glob("*.json")).read_text(encoding="utf-8"))
    assert {rec["model_a"], rec["model_b"]} == {
        "groq/openai/gpt-oss-120b", "google/gemini-3.5-flash"}
    assert rec["elicited_via"] == "html"
    assert rec["grader"] == "alice"


def test_ingest_is_idempotent(bundle_and_votes):
    """Re-sending the same file must not double-count a vote."""
    _, votes_path, tmp_path = bundle_and_votes
    pref = tmp_path / "pref"
    first = iv.ingest_one(str(votes_path), str(pref))
    second = iv.ingest_one(str(votes_path), str(pref))
    assert first and second == []


def test_two_graders_on_one_bundle_produce_separate_records(bundle_and_votes):
    """Inter-rater disagreement is data; one grader must not overwrite another."""
    bundle, votes_path, tmp_path = bundle_and_votes
    pref = tmp_path / "pref"
    iv.ingest_one(str(votes_path), str(pref))
    other = json.loads(votes_path.read_text(encoding="utf-8"))
    other["grader"] = "bob"
    other_path = tmp_path / "votes_bob.json"
    other_path.write_text(json.dumps(other), encoding="utf-8")
    iv.ingest_one(str(other_path), str(pref))
    names = sorted(p.name for p in pref.glob("*.json"))
    assert any("alice" in n for n in names) and any("bob" in n for n in names)
    assert len(names) == 2 * len(bundle["tasks"])


def test_ingest_carries_the_note_through(bundle_and_votes):
    bundle, votes_path, tmp_path = bundle_and_votes
    pref = tmp_path / "pref"
    iv.ingest_one(str(votes_path), str(pref))
    first_task = bundle["tasks"][0]["task_id"]
    rec = json.loads(next(pref.glob(f"{first_task}__*.json")).read_text(encoding="utf-8"))
    assert rec["note"] == "because"


def test_ingest_dry_run_writes_nothing(bundle_and_votes):
    _, votes_path, tmp_path = bundle_and_votes
    pref = tmp_path / "pref"
    iv.ingest_one(str(votes_path), str(pref), dry_run=True)
    assert not pref.exists() or not list(pref.glob("*.json"))


def test_ingest_refuses_an_unknown_bundle(tmp_path):
    """Without the mapping the votes are unresolvable, and silently guessing
    would attach a verdict to the wrong model."""
    p = tmp_path / "v.json"
    p.write_text(json.dumps({"bundle_id": "deadbeef0000", "grader": "x",
                             "votes": {}}), encoding="utf-8")
    with pytest.raises(SystemExit):
        iv.ingest_one(str(p), str(tmp_path / "pref"))


def test_ingest_rejects_a_file_that_is_not_a_votes_file(tmp_path):
    p = tmp_path / "v.json"
    p.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    with pytest.raises(SystemExit):
        iv.ingest_one(str(p), str(tmp_path / "pref"))
