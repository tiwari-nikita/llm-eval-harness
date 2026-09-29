import json
from collections import Counter
from itertools import combinations
from unittest.mock import patch

import pytest

import replay as rp
from runner import DailyQuotaExceeded


@pytest.fixture(autouse=True)
def isolated_dir(tmp_path, monkeypatch):
    """Keep every test out of the real prompts/replay/, which holds real data."""
    d = tmp_path / "replay"
    monkeypatch.setattr(rp, "REPLAY_DIR", d)
    return d


def _node(nid, parent, role, parts, content_type="text", **meta):
    return {"id": nid, "parent": parent, "message": {
        "author": {"role": role}, "create_time": 0,
        "content": {"content_type": content_type, "parts": parts},
        "metadata": meta}}


def _conv(cid, title, text, images=0, attachments=0):
    parts = [text] + [{"asset_pointer": "x"}] * images
    ctype = "multimodal_text" if images else "text"
    mapping = {
        "root": {"id": "root", "parent": None, "message": None},
        "u1": _node("u1", "root", "user", parts, ctype,
                    attachments=[{"id": "f"}] * attachments),
        "a1": _node("a1", "u1", "assistant", ["reply"]),
    }
    return {"conversation_id": cid, "title": title, "mapping": mapping,
            "current_node": "a1"}


PROVIDERS = {
    "groq": {"base_url": "https://g", "api_key_env": "GROQ_API_KEY",
             "models": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]},
    "google": {"base_url": "https://o", "api_key_env": "GOOGLE_API_KEY",
               "models": ["gemini-3.5-flash", "gemini-3.5-flash-lite"]},
}


# ---------------------------------------------------------------- the export

def test_opening_follows_the_branch_actually_used_not_an_abandoned_edit():
    conv = _conv("c1", "t", "final version")
    # an edited-away first message, still present in the tree
    conv["mapping"]["u0"] = _node("u0", "root", "user", ["abandoned draft"])
    [item] = rp.opening_prompts([conv])
    assert item["text"] == "final version"


def test_opening_skips_hidden_and_non_text_user_messages():
    conv = _conv("c1", "t", "real opening")
    conv["mapping"]["h"] = _node("h", "root", "user", ["system context"],
                                 is_visually_hidden_from_conversation=True)
    conv["mapping"]["u1"]["parent"] = "h"
    [item] = rp.opening_prompts([conv])
    assert item["text"] == "real opening"


def test_opening_counts_images_and_attachments():
    [item] = rp.opening_prompts([_conv("c1", "t", "caption this", images=2, attachments=1)])
    assert item["n_images"] == 2 and item["n_attachments"] == 1


def test_opening_without_current_node_falls_back_to_time_order():
    conv = _conv("c1", "t", "hello there")
    del conv["current_node"]
    [item] = rp.opening_prompts([conv])
    assert item["text"] == "hello there"


def test_opening_id_is_stable_and_does_not_leak_the_conversation_id():
    [a] = rp.opening_prompts([_conv("secret-id", "t", "x y z")])
    [b] = rp.opening_prompts([_conv("secret-id", "t", "x y z")])
    assert a["id"] == b["id"] and "secret" not in a["id"]


# ---------------------------------------------------------------- topics and flags

@pytest.mark.parametrize("title,text,topic", [
    ("BTC", "turn this into a tweet about btc hitting 100k", "social"),
    ("x", "what does mahadasha of venus mean", "astrology"),
    ("x", "elusive meaning", "explain"),
    ("x", "roof parapet", "explain"),
    ("x", "what’s better beige and white or beige and khaki", "advice"),
    ("x", "of course i want a long detailed plan for the trip to goa and back", "other"),
    ("x", "rewrite this so it sounds more intellectual please thanks", "writing"),
])
def test_classify_topic(title, text, topic):
    assert rp.classify_topic(title, text) == topic


@pytest.mark.parametrize("text,reason", [
    ("", "no text"),
    ("a tweet caption for this - i have attached a picture", "refers to something not in the text"),
    ("create an illustration of my pet napping by the fire", "asks for an image"),
])
def test_unusable_reasons(text, reason):
    item = {"text": text, "words": len(text.split()), "n_images": 0, "n_attachments": 0}
    assert rp.unusable_reason(item) == reason


def test_emotionally_attached_is_not_mistaken_for_an_attachment():
    text = "i am so attached to this idea of moving to new york after graduation"
    item = {"text": text, "words": len(text.split()), "n_images": 0, "n_attachments": 0}
    assert rp.unusable_reason(item) is None


@pytest.mark.parametrize("text,flag", [
    ("should i forgive my ex for what he said", "mentions a relationship"),
    ("born 14 march 1999 at 4:35 am in pune, what is my moon sign", "may contain birth details"),
    ("diet to go from 72kgs to 58kgs on 1400 cal", "mentions health or body"),
    ("when will i get my passport back after visa stamping", "mentions visa or immigration"),
    ("mail me at someone@example.com with the plan", "may contain contact details"),
    ("call me on +1 617 555 0199 tomorrow", "may contain a phone or ID number"),
])
def test_private_flags(text, flag):
    item = {"title": "", "text": text, "words": len(text.split())}
    assert flag in rp.private_flags(item)


def test_clean_prompt_has_no_flags():
    item = {"title": "Web3 tweets", "text": "give me tweet ideas for web3 marketing", "words": 7}
    assert rp.private_flags(item) == []


def test_long_paste_is_flagged():
    text = "word " * (rp.LONG_WORDS + 1)
    assert any(f.startswith("long paste") for f in
               rp.private_flags({"title": "", "text": text, "words": rp.LONG_WORDS + 1}))


# ---------------------------------------------------------------- sampling

def test_allocate_sums_to_total_and_respects_availability():
    alloc = rp.allocate({"a": 100, "b": 30, "c": 4}, 50)
    assert sum(alloc.values()) == 50
    assert alloc["c"] == 4
    assert alloc["a"] > alloc["b"] > alloc["c"]


def test_allocate_never_exceeds_what_exists():
    alloc = rp.allocate({"a": 3, "b": 2}, 50)
    assert alloc == {"a": 3, "b": 2}


def test_allocate_floor_shrinks_when_total_is_small():
    alloc = rp.allocate({t: 100 for t in "abcdef"}, 12)
    assert sum(alloc.values()) == 12 and min(alloc.values()) >= 1


def _openings(n_clean=40, n_private=10):
    convs = [_conv(f"c{i}", "Tweet ideas", f"give me tweet ideas number {i} for web3")
             for i in range(n_clean)]
    convs += [_conv(f"p{i}", "Ex", f"should i text my ex again, attempt {i}")
              for i in range(n_private)]
    convs += [_conv("img", "Caption", "caption this", images=1)]
    return rp.opening_prompts(convs)


def test_build_sample_ticks_clean_and_offers_private_unticked():
    items, dropped, n_usable = rp.build_sample(_openings(), n=20, n_optin=5, seed=1)
    on = [it for it in items if it["default"]]
    off = [it for it in items if not it["default"]]
    assert len(on) == 20 and len(off) == 5
    assert all(it["flags"] for it in off) and not any(it["flags"] for it in on)
    assert dropped["had an image or file attached"] == 1
    assert n_usable == 50


def test_build_sample_is_deterministic_for_a_seed():
    a, _, _ = rp.build_sample(_openings(), n=20, n_optin=5, seed=7)
    b, _, _ = rp.build_sample(_openings(), n=20, n_optin=5, seed=7)
    assert [it["id"] for it in a] == [it["id"] for it in b]


def _write_sample(tmp_path, items, providers=("groq", "google"), seed=3):
    rp.REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    sample = {"sample_id": "s1", "seed": seed, "providers": list(providers),
              "system": rp.REPLAY_SYSTEM, "topics": rp.TOPIC_LABELS, "items": items}
    (rp.REPLAY_DIR / "sample.json").write_text(json.dumps(sample), encoding="utf-8")
    return sample


def _items(n, topic="social"):
    return [{"id": f"p_{i:03d}", "title": "t", "text": f"prompt number {i}",
             "words": 3, "topic": topic, "flags": [], "default": True} for i in range(n)]


def test_cmd_sample_writes_review_page_and_sends_nothing(tmp_path, monkeypatch):
    export = tmp_path / "conversations.json"
    export.write_text(json.dumps([_conv(f"c{i}", "Tweets", f"tweet ideas about topic {i}")
                                  for i in range(12)]), encoding="utf-8")
    providers_file = tmp_path / "providers.yaml"
    import yaml
    providers_file.write_text(yaml.safe_dump(PROVIDERS), encoding="utf-8")

    class A:
        pass
    args = A()
    args.export, args.providers, args.providers_file = str(export), "groq,google", str(providers_file)
    args.n, args.optin, args.seed = 10, 2, 5
    with patch("runner.requests.post") as post:
        rp.cmd_sample(args)
        post.assert_not_called()
    html = (rp.REPLAY_DIR / "review.html").read_text(encoding="utf-8")
    assert "/*__SAMPLE__*/null" not in html
    assert "Google AI Studio" in html  # the data-use note reaches the page


def test_review_page_escapes_script_close_in_prompt_text(tmp_path):
    html = rp._inline(rp.REVIEW_TEMPLATE, "/*__SAMPLE__*/null",
                      {"items": [{"text": "</script><b>x</b>"}]})
    assert "</script><b>" not in html


# ---------------------------------------------------------------- approval

def _approval(tmp_path, **kw):
    body = {"sample_id": "s1", "providers": ["groq", "google"],
            "include": ["p_000", "p_001"], "topics": {}}
    body.update(kw)
    path = tmp_path / "approved.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_approval_for_another_sample_is_refused(tmp_path):
    sample = _write_sample(tmp_path, _items(3))
    with pytest.raises(SystemExit, match="current sample"):
        rp.load_approval(sample, _approval(tmp_path, sample_id="other"))


def test_approval_with_unknown_prompt_id_is_refused(tmp_path):
    sample = _write_sample(tmp_path, _items(3))
    with pytest.raises(SystemExit, match="not in"):
        rp.load_approval(sample, _approval(tmp_path, include=["p_000", "p_999"]))


def test_approval_naming_an_unshown_provider_is_refused(tmp_path):
    sample = _write_sample(tmp_path, _items(3), providers=("groq",))
    with pytest.raises(SystemExit, match="never showed"):
        rp.load_approval(sample, _approval(tmp_path, providers=["groq", "openrouter"]))


def test_approval_applies_topic_overrides(tmp_path):
    sample = _write_sample(tmp_path, _items(3))
    items, _ = rp.load_approval(sample, _approval(tmp_path, topics={"p_001": "career"}))
    assert {it["id"]: it["topic"] for it in items}["p_001"] == "career"


# ---------------------------------------------------------------- pairing

def test_assign_pairs_is_balanced_overall_and_per_topic():
    items = _items(30, "social") + [dict(it, id="q" + it["id"], topic="crypto")
                                    for it in _items(18)]
    models = rp.model_list(PROVIDERS, ["groq", "google"])
    pairs = rp.assign_pairs(items, models, seed=1)
    overall = Counter(pairs.values())
    assert len(overall) == 6 and max(overall.values()) - min(overall.values()) <= 1
    for topic in ("social", "crypto"):
        c = Counter(pairs[it["id"]] for it in items if it["topic"] == topic)
        assert max(c.values()) - min(c.values()) <= 1


def test_assign_pairs_is_deterministic():
    models = rp.model_list(PROVIDERS, ["groq", "google"])
    assert rp.assign_pairs(_items(12), models, 4) == rp.assign_pairs(_items(12), models, 4)


# ---------------------------------------------------------------- running

class RunArgs:
    def __init__(self, approved, providers_file, dry_run=False):
        self.approved, self.providers_file, self.dry_run = str(approved), str(providers_file), dry_run
        self.max_tokens, self.tpm, self.gap = 512, 100000, 0


@pytest.fixture
def providers_file(tmp_path):
    import yaml
    p = tmp_path / "providers.yaml"
    p.write_text(yaml.safe_dump(PROVIDERS), encoding="utf-8")
    return p


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k1")
    monkeypatch.setenv("GOOGLE_API_KEY", "k2")
    monkeypatch.setattr(rp, "load_env_file", lambda path: None)


def _fake_reply(content="an answer"):
    return {"content": content, "finish_reason": "stop", "completion_tokens": 5,
            "total_tokens": 20}


def test_run_sends_only_approved_prompts(tmp_path, providers_file, keys):
    items = _items(4)
    items[3]["text"] = "PRIVATE unapproved text"
    _write_sample(tmp_path, items)
    approved = _approval(tmp_path, include=["p_000", "p_001", "p_002"])
    with patch.object(rp, "call_model", return_value=_fake_reply()) as cm:
        rp.cmd_run(RunArgs(approved, providers_file))
    sent = [c.args[3][-1]["content"] for c in cm.call_args_list]
    assert len(sent) == 6  # three prompts, two models each
    assert "PRIVATE unapproved text" not in sent
    assert all(c.args[3][0] == {"role": "system", "content": rp.REPLAY_SYSTEM}
               for c in cm.call_args_list)


def test_dry_run_sends_nothing(tmp_path, providers_file, keys):
    _write_sample(tmp_path, _items(3))
    with patch.object(rp, "call_model") as cm:
        rp.cmd_run(RunArgs(_approval(tmp_path), providers_file, dry_run=True))
    cm.assert_not_called()
    assert not (rp.REPLAY_DIR / "approved.json").exists()


def test_run_refuses_when_an_approved_provider_has_no_key(tmp_path, providers_file, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "k1")
    monkeypatch.setattr(rp, "load_env_file", lambda path: None)
    _write_sample(tmp_path, _items(3))
    with patch.object(rp, "call_model") as cm, pytest.raises(SystemExit, match="GOOGLE_API_KEY"):
        rp.cmd_run(RunArgs(_approval(tmp_path), providers_file))
    cm.assert_not_called()


def test_run_resumes_without_resending_saved_answers(tmp_path, providers_file, keys):
    _write_sample(tmp_path, _items(3))
    approved = _approval(tmp_path)
    with patch.object(rp, "call_model", return_value=_fake_reply()) as cm:
        rp.cmd_run(RunArgs(approved, providers_file))
        first = cm.call_count
        rp.cmd_run(RunArgs(approved, providers_file))
    assert first == 4 and cm.call_count == 4


def test_run_retries_error_records_on_the_next_run(tmp_path, providers_file, keys):
    _write_sample(tmp_path, _items(3))
    approved = _approval(tmp_path)
    with patch.object(rp, "call_model", side_effect=RuntimeError("boom")):
        rp.cmd_run(RunArgs(approved, providers_file))
    with patch.object(rp, "call_model", return_value=_fake_reply()) as cm:
        rp.cmd_run(RunArgs(approved, providers_file))
    assert cm.call_count == 4


def test_one_models_daily_quota_does_not_stop_the_others(tmp_path, providers_file, keys):
    _write_sample(tmp_path, _items(12))
    approved = _approval(tmp_path, include=[f"p_{i:03d}" for i in range(12)])

    def fake(base_url, key, model, messages, **kw):
        if model == "gemini-3.5-flash":
            raise DailyQuotaExceeded("gone")
        return _fake_reply()
    with patch.object(rp, "call_model", side_effect=fake) as cm:
        rp.cmd_run(RunArgs(approved, providers_file))
    models_called = Counter(c.args[2] for c in cm.call_args_list)
    assert models_called["gemini-3.5-flash"] == 1  # tried once, then skipped
    assert models_called["openai/gpt-oss-120b"] > 0


# ---------------------------------------------------------------- the pick page

def _run_all(tmp_path, providers_file, n=6):
    _write_sample(tmp_path, _items(n))
    approved = _approval(tmp_path, include=[f"p_{i:03d}" for i in range(n)])
    with patch.object(rp, "call_model",
                      side_effect=lambda b, k, m, msgs, **kw: _fake_reply("words " * (3 if "20b" in m else 9))):
        rp.cmd_run(RunArgs(approved, providers_file))
    return approved


def _page_bundle():
    html = (rp.REPLAY_DIR / "pick.html").read_text(encoding="utf-8")
    start = html.index("const BUNDLE = ") + len("const BUNDLE = ")
    return json.loads(html[start:html.index(";\n", start)].replace("<\\/", "</")), html


def test_pick_page_holds_no_model_or_provider_names(tmp_path, providers_file, keys):
    _run_all(tmp_path, providers_file)
    _, html = _page_bundle()
    for name in ("gpt-oss", "gemini", "groq", "google"):
        assert name not in html.lower()


def test_pick_page_is_stable_across_rebuilds(tmp_path, providers_file, keys):
    approved = _run_all(tmp_path, providers_file)
    b1, _ = _page_bundle()
    with patch.object(rp, "call_model") as cm:
        rp.cmd_run(RunArgs(approved, providers_file))
    cm.assert_not_called()
    b2, _ = _page_bundle()
    assert b1 == b2


def test_pick_page_leaves_out_prompts_missing_an_answer(tmp_path, providers_file, keys):
    _write_sample(tmp_path, _items(3))
    approved = _approval(tmp_path, include=["p_000", "p_001", "p_002"])

    def fake(base_url, key, model, messages, **kw):
        if messages[-1]["content"] == "prompt number 0" and "gemini" in model:
            raise RuntimeError("down")
        return _fake_reply()
    with patch.object(rp, "call_model", side_effect=fake):
        rp.cmd_run(RunArgs(approved, providers_file))
    bundle, _ = _page_bundle()
    ids = {it["id"] for it in bundle["items"]}
    pairs = rp.assign_pairs(_items(3), rp.model_list(PROVIDERS, ["groq", "google"]), 3)
    needs_gemini = any("gemini" in m for _, m in pairs["p_000"])
    assert ("p_000" in ids) != needs_gemini


# ---------------------------------------------------------------- the card

def _picks_file(tmp_path, bundle_id, picks):
    p = tmp_path / f"picks_{bundle_id}.json"
    p.write_text(json.dumps({"bundle_id": bundle_id, "picks": picks}), encoding="utf-8")
    return p


def test_ingest_resolves_sides_to_models(tmp_path, providers_file, keys):
    _run_all(tmp_path, providers_file)
    bundle, _ = _page_bundle()
    state = json.loads((rp.REPLAY_DIR / f"pick_state_{bundle['bundle_id']}.json").read_text())
    first = bundle["items"][0]["id"]
    store = rp.ingest([_picks_file(tmp_path, bundle["bundle_id"],
                                   {first: {"choice": "b", "tags": ["right tone"]}})])
    side = state["mapping"][first]["b"]
    assert store[first]["model_b"] == f"{side['provider']}/{side['model']}"
    assert store[first]["choice"] == "b" and store[first]["tags"] == ["right tone"]


def test_ingest_refuses_picks_for_an_unknown_bundle(tmp_path):
    with pytest.raises(SystemExit, match="no local mapping"):
        rp.ingest([_picks_file(tmp_path, "nope", {})])


def test_ingest_ignores_invalid_choices(tmp_path, providers_file, keys):
    _run_all(tmp_path, providers_file)
    bundle, _ = _page_bundle()
    first = bundle["items"][0]["id"]
    store = rp.ingest([_picks_file(tmp_path, bundle["bundle_id"], {first: {"choice": "maybe"}})])
    assert first not in store


def _records(outcomes, topic="social"):
    """outcomes: list of (model_a, model_b, choice)."""
    return [{"model_a": a, "model_b": b, "choice": c, "topic": topic, "tags": [],
             "a_words": 10, "b_words": 10} for a, b, c in outcomes]


MODELS = ["m1", "m2", "m3", "m4"]


def test_a_model_that_wins_every_matchup_is_a_clear_pick():
    outcomes = []
    for a, b in combinations(MODELS, 2):
        winner = "a" if a == "m1" else ("b" if b == "m1" else "tie")
        outcomes += [(a, b, winner)] * 4
    v = rp.topic_verdict(_records(outcomes), MODELS)
    assert v["leader"] == "m1" and v["verdict"] == "clear"


def test_all_ties_cannot_be_told_apart():
    outcomes = [(a, b, "tie") for a, b in combinations(MODELS, 2)] * 4
    v = rp.topic_verdict(_records(outcomes), MODELS)
    assert v["verdict"] == "too few picks"
    assert v["undecided"] == v["n"]


def test_a_handful_of_picks_is_too_few():
    v = rp.topic_verdict(_records([("m1", "m2", "a")] * 3), MODELS)
    assert v["verdict"] == "too few picks"


def test_bradley_terry_orders_by_strength():
    votes = [("m1", "m2", 1.0)] * 6 + [("m2", "m3", 1.0)] * 6 + [("m1", "m3", 1.0)] * 6
    s = rp.bradley_terry(votes, ["m1", "m2", "m3"])
    assert s["m1"] > s["m2"] > s["m3"]


def test_bias_checks_count_left_and_longer_picks():
    recs = _records([("m1", "m2", "a"), ("m1", "m2", "a"), ("m1", "m2", "b"), ("m1", "m2", "tie")])
    recs[0]["a_words"] = 50   # left and longer, picked
    recs[2]["b_words"] = 50   # right and longer, picked
    b = rp.bias_checks(recs)
    assert (b["decisive"], b["left"]) == (3, 2)
    assert (b["sized"], b["longer"]) == (2, 2)


def test_render_card_names_the_leader_and_reports_undecided():
    outcomes = []
    for a, b in combinations(MODELS, 2):
        winner = "a" if a == "m1" else ("b" if b == "m1" else "tie")
        outcomes += [(a, b, winner)] * 4
    store = {str(i): r for i, r in enumerate(_records(
        [(f"groq/openai/{a}", f"groq/openai/{b}", c) for a, b, c in outcomes]))}
    card = rp.render_card(store)
    assert "Tweets, replies and LinkedIn" in card and "m1" in card and "clear" in card


def test_wilson_interval_brackets_the_rate():
    lo, hi = rp.wilson(30, 60)
    assert lo < 0.5 < hi


# ---------------------------------------------------------------- .env

def test_load_env_file_fills_missing_without_overriding(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nGROQ_API_KEY="from-file"\nexport GOOGLE_API_KEY=g\n', encoding="utf-8")
    monkeypatch.setenv("GROQ_API_KEY", "already-set")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    rp.load_env_file(env)
    import os
    assert os.environ["GROQ_API_KEY"] == "already-set"
    assert os.environ["GOOGLE_API_KEY"] == "g"


# ---------------------------------------------------------------- judges

class JudgeArgs:
    def __init__(self, providers_file, judge=None, dry_run=False):
        self.judge, self.providers_file, self.dry_run = judge, str(providers_file), dry_run
        self.max_tokens, self.tpm, self.gap = 3000, 100000, 0


@pytest.mark.parametrize("text,choice", [
    ("B", "b"),
    ("A", "a"),
    ("TIE", "tie"),
    ("Answer A hedges; B commits to a plan.\n\nB", "b"),
    ("Both are fine, so: tie", "tie"),
    ("a thoughtful answer either way", None),
    ("", None),
])
def test_parse_judgment_takes_the_last_verdict(text, choice):
    assert rp.parse_judgment(text) == choice


def test_judge_refuses_a_provider_you_did_not_approve(tmp_path, providers_file, keys):
    _run_all(tmp_path, providers_file)
    approved = json.loads((rp.REPLAY_DIR / "approved.json").read_text())
    approved["providers"] = ["groq"]
    (rp.REPLAY_DIR / "approved.json").write_text(json.dumps(approved))
    with patch.object(rp, "call_model") as cm, pytest.raises(SystemExit, match="not approved"):
        rp.cmd_judge(JudgeArgs(providers_file, judge=["google:gemini-3.6-flash"]))
    cm.assert_not_called()


def test_judge_sees_the_same_answers_on_the_same_sides(tmp_path, providers_file, keys):
    _run_all(tmp_path, providers_file)
    bundle, _ = _page_bundle()
    with patch.object(rp, "call_model", return_value=_fake_reply("A")) as cm:
        rp.cmd_judge(JudgeArgs(providers_file, judge=["groq:openai/gpt-oss-120b"]))
    assert cm.call_count == len(bundle["items"])
    sent = [c.args[3][0]["content"] for c in cm.call_args_list]
    for it in bundle["items"]:
        assert any(f"Answer A:\n---\n{it['a']}\n---\n\nAnswer B:\n---\n{it['b']}\n" in s
                   and it["prompt"] in s for s in sent)


def test_judge_never_sees_an_unapproved_prompt(tmp_path, providers_file, keys):
    items = _items(4)
    items[3]["text"] = "PRIVATE unapproved text"
    _write_sample(tmp_path, items)
    approved = _approval(tmp_path, include=["p_000", "p_001", "p_002"])
    with patch.object(rp, "call_model", return_value=_fake_reply()):
        rp.cmd_run(RunArgs(approved, providers_file))
    with patch.object(rp, "call_model", return_value=_fake_reply("B")) as cm:
        rp.cmd_judge(JudgeArgs(providers_file, judge=["groq:openai/gpt-oss-120b"]))
    assert not any("PRIVATE" in c.args[3][0]["content"] for c in cm.call_args_list)


def test_judge_resumes_and_retries_a_cut_off_verdict(tmp_path, providers_file, keys):
    _run_all(tmp_path, providers_file)
    cut = dict(_fake_reply("Weighing A against"), finish_reason="length")
    with patch.object(rp, "call_model", side_effect=[cut] + [_fake_reply("B")] * 20) as cm:
        rp.cmd_judge(JudgeArgs(providers_file, judge=["groq:openai/gpt-oss-120b"]))
        first = cm.call_count
        rp.cmd_judge(JudgeArgs(providers_file, judge=["groq:openai/gpt-oss-120b"]))
    assert cm.call_count == first + 1  # only the cut-off one is asked again
    verdicts = rp.load_judges()["groq/openai/gpt-oss-120b"]
    assert all(v["choice"] == "b" for v in verdicts.values())


def test_one_judges_daily_quota_does_not_stop_the_other(tmp_path, providers_file, keys):
    _run_all(tmp_path, providers_file)

    def fake(base_url, key, model, messages, **kw):
        if model == "gemini-3.6-flash":
            raise DailyQuotaExceeded("gone")
        return _fake_reply("A")
    with patch.object(rp, "call_model", side_effect=fake):
        rp.cmd_judge(JudgeArgs(providers_file))
    judges = rp.load_judges()
    assert len(judges["groq/openai/gpt-oss-120b"]) == 6
    assert not judges.get("google/gemini-3.6-flash")


def test_judge_dry_run_sends_nothing(tmp_path, providers_file, keys):
    _run_all(tmp_path, providers_file)
    with patch.object(rp, "call_model") as cm:
        rp.cmd_judge(JudgeArgs(providers_file, dry_run=True))
    cm.assert_not_called()


def _judged_store(n=40):
    """Pairs of gpt-oss-20b vs gemini; you always pick gemini, on either side."""
    store, agreeing, own_family = {}, {}, {}
    for i in range(n):
        gpt_left = i % 2 == 0
        a, b = ("groq/openai/gpt-oss-20b", "google/gemini-3.5-flash")[::1 if gpt_left else -1]
        yours = "b" if gpt_left else "a"
        store[f"p{i}"] = {"model_a": a, "model_b": b, "choice": yours, "topic": "social",
                          "tags": [], "a_words": 10, "b_words": 10}
        agreeing[f"p{i}"] = {"choice": yours}
        own_family[f"p{i}"] = {"choice": "a" if gpt_left else "b"}
    return store, agreeing, own_family


def test_render_judges_calls_a_judge_that_matches_you_a_stand_in():
    store, agreeing, _ = _judged_store()
    text = "\n".join(rp.render_judges(store, {"google/gemini-3.6-flash": agreeing}, {}))
    assert "agrees with your pick on 100% of 40" in text
    assert "close enough to stand in for you" in text


def test_render_judges_exposes_self_family_preference():
    store, _, own_family = _judged_store()
    text = "\n".join(rp.render_judges(store, {"groq/openai/gpt-oss-120b": own_family}, {}))
    assert "agrees with your pick on 0% of 40" in text
    assert "disagrees with you too often" in text
    assert "picks its own family's answer 100% of 40 times (you, same pairs: 0%)" in text


def test_render_judges_counts_unreadable_verdicts():
    store, agreeing, _ = _judged_store(10)
    agreeing["p0"] = {"choice": None}
    text = "\n".join(rp.render_judges(store, {"google/gemini-3.6-flash": agreeing}, {}))
    assert "1 verdict(s) it gave could not be read" in text


def test_card_includes_the_judge_section_when_judges_exist():
    store, agreeing, _ = _judged_store()
    card = rp.render_card(store, {"google/gemini-3.6-flash": agreeing})
    assert "LLM judges on your pairs" in card
    assert "same top pick as you in 1 of 1 topic(s)" in card


def test_judge_obeys_a_narrowed_approval_over_an_older_pick_page(tmp_path, providers_file, keys):
    """The pick page can predate the current approval, e.g. if you narrowed it
    and the run then stopped. The approval, not the page, decides what goes out."""
    items = _items(3)
    items[2]["text"] = "PRIVATE later unticked"
    _write_sample(tmp_path, items)
    with patch.object(rp, "call_model", return_value=_fake_reply()):
        rp.cmd_run(RunArgs(_approval(tmp_path, include=["p_000", "p_001", "p_002"]), providers_file))
    narrowed = json.loads((rp.REPLAY_DIR / "approved.json").read_text())
    narrowed["include"] = ["p_000", "p_001"]
    (rp.REPLAY_DIR / "approved.json").write_text(json.dumps(narrowed))
    with patch.object(rp, "call_model", return_value=_fake_reply("A")) as cm:
        rp.cmd_judge(JudgeArgs(providers_file, judge=["groq:openai/gpt-oss-120b"]))
    assert cm.call_count == 2
    assert not any("PRIVATE" in c.args[3][0]["content"] for c in cm.call_args_list)


def test_run_that_stops_on_a_missing_key_records_no_approval(tmp_path, providers_file, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "k1")
    monkeypatch.setattr(rp, "load_env_file", lambda path: None)
    _write_sample(tmp_path, _items(3))
    with pytest.raises(SystemExit):
        rp.cmd_run(RunArgs(_approval(tmp_path), providers_file))
    assert not (rp.REPLAY_DIR / "approved.json").exists()
