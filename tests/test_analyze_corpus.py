from analyze_corpus import user_turns, profile


def test_user_turns_conversation_key():
    rec = {"conversation": [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "human", "content": "second turn"},
    ]}
    assert user_turns(rec) == ["hello", "second turn"]


def test_user_turns_content_as_list_of_dicts():
    rec = {"messages": [{"from": "user", "value": [{"text": "part a"}, {"text": "part b"}]}]}
    assert user_turns(rec) == ["part a part b"]


def test_user_turns_no_recognized_key_returns_empty():
    assert user_turns({"unrelated": 1}) == []


def test_profile_buckets_and_turn_counts():
    records = [
        {"conversation": [{"role": "user", "content": "write a poem about the sea"}]},
        {"conversation": [
            {"role": "user", "content": "debug this python function"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "still broken"},
        ]},
    ]
    p = profile(records, sample_n=1)
    assert p["convos"] == 2
    assert p["avg_turns"] == 1.5
    assert p["buckets"]["writing"] == 1
    assert p["buckets"]["code"] == 1
    assert p["turn_counts"][1] == 1
    assert p["turn_counts"][2] == 1


def test_profile_skips_records_with_no_user_turns():
    records = [{"unrelated": 1}, {"conversation": [{"role": "user", "content": "hi there friend"}]}]
    p = profile(records)
    assert p["convos"] == 1
