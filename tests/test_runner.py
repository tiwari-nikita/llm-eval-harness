from unittest.mock import patch, MagicMock

from runner import call_model, run_one, safe_slug


def _resp(status_code, content=None):
    r = MagicMock()
    r.status_code = status_code
    if status_code < 400:
        r.json.return_value = {"choices": [{"message": {"content": content}}]}
        r.raise_for_status.return_value = None
    else:
        r.raise_for_status.side_effect = Exception(f"http {status_code}")
    return r


def test_safe_slug():
    assert safe_slug("meta-llama/llama-3.1-8b-instruct:free") == "meta-llama_llama-3.1-8b-instruct_free"


@patch("runner.time.sleep", return_value=None)
@patch("runner.requests.post")
def test_call_model_retries_on_429_then_succeeds(mock_post, _sleep):
    mock_post.side_effect = [_resp(429), _resp(200, "final answer")]
    out = call_model("http://fake", "key", "model", [{"role": "user", "content": "hi"}])
    assert out == "final answer"
    assert mock_post.call_count == 2


@patch("runner.requests.post")
def test_call_model_single_success(mock_post):
    mock_post.return_value = _resp(200, "hello")
    out = call_model("http://fake", "key", "model", [{"role": "user", "content": "hi"}])
    assert out == "hello"


@patch("runner.call_model")
def test_run_one_builds_growing_message_history(mock_call_model):
    # each call returns a distinct reply so we can check history grew correctly
    replies = iter(["reply1", "reply2", "reply3"])
    mock_call_model.side_effect = lambda base_url, api_key, model, messages: next(replies)

    task = {"id": "advisory_001", "turns": ["turn a", "turn b", "turn c"]}
    result = run_one(task, "groq", {"base_url": "http://x"}, "model-x", "key", 0)

    assert result["multiturn"] is True
    assert len(result["exchanges"]) == 3
    assert [e["assistant"] for e in result["exchanges"]] == ["reply1", "reply2", "reply3"]

    # the third call's message list must contain all prior turns
    third_call_messages = mock_call_model.call_args_list[2].args[3]
    user_turns = [m["content"] for m in third_call_messages if m["role"] == "user"]
    assert user_turns == ["turn a", "turn b", "turn c"]


@patch("runner.call_model")
def test_run_one_single_turn_task_uses_prompt(mock_call_model):
    mock_call_model.return_value = "the answer"
    task = {"id": "research_001", "prompt": "what is x"}
    result = run_one(task, "groq", {"base_url": "http://x"}, "model-x", "key", 0)

    assert result["multiturn"] is False
    assert len(result["exchanges"]) == 1
    assert result["exchanges"][0]["user"] == "what is x"
    assert result["exchanges"][0]["assistant"] == "the answer"
