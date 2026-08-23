from extract_prompts import role_of, text_of, walk


def test_role_of_simple_string():
    assert role_of({"role": "user"}) == "user"


def test_role_of_nested_author_dict():
    assert role_of({"author": {"role": "user"}}) == "user"


def test_role_of_missing():
    assert role_of({"foo": "bar"}) is None


def test_text_of_plain_string():
    assert text_of({"text": "hello there"}) == "hello there"


def test_text_of_content_list_of_blocks():
    obj = {"content": [{"text": "part one"}, {"text": "part two"}]}
    assert text_of(obj) == "part one\npart two"


def test_text_of_chatgpt_parts_shape():
    obj = {"content": {"parts": ["hi", "there"]}}
    assert text_of(obj) == "hi\nthere"


def test_text_of_blank_returns_none():
    assert text_of({"text": "   "}) is None
    assert text_of({}) is None


def test_walk_claude_style_export():
    data = [
        {
            "name": "convo A",
            "chat_messages": [
                {"sender": "human", "text": "first human message"},
                {"sender": "assistant", "text": "a reply, ignored"},
                {"sender": "human", "text": "second human message"},
            ],
        }
    ]
    out = []
    walk(data, out)
    texts = [m["text"] for m in out]
    assert texts == ["first human message", "second human message"]
    assert all(m["conversation"] == "convo A" for m in out)


def test_walk_chatgpt_style_export():
    data = {
        "title": "convo B",
        "mapping": {
            "id1": {"message": {"author": {"role": "user"},
                                 "content": {"parts": ["chatgpt style message"]}}},
            "id2": {"message": {"author": {"role": "assistant"},
                                 "content": {"parts": ["ignored reply"]}}},
        },
    }
    out = []
    walk(data, out)
    assert len(out) == 1
    assert out[0]["text"] == "chatgpt style message"
    assert out[0]["conversation"] == "convo B"


def test_walk_ignores_non_human_roles():
    data = [{"role": "assistant", "text": "should not be collected"}]
    out = []
    walk(data, out)
    assert out == []
