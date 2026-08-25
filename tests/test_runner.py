import pytest
from unittest.mock import patch, MagicMock

from runner import (TokenBudget, call_model, estimate_tokens, parse_duration,
                    run_one, safe_slug)


def _resp(status_code, content=None, finish_reason="stop", total_tokens=None,
          headers=None):
    r = MagicMock()
    r.status_code = status_code
    r.headers = headers or {}
    r.text = "body"
    if status_code < 400:
        r.json.return_value = {
            "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
            "usage": {"completion_tokens": 7, "total_tokens": total_tokens or 20},
        }
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
    assert out["content"] == "final answer"
    assert mock_post.call_count == 2


@patch("runner.requests.post")
def test_call_model_single_success(mock_post):
    mock_post.return_value = _resp(200, "hello")
    out = call_model("http://fake", "key", "model", [{"role": "user", "content": "hi"}])
    assert out["content"] == "hello"
    assert out["finish_reason"] == "stop"


@patch("runner.requests.post")
def test_call_model_reports_length_finish_reason(mock_post):
    """A harness-truncated reply must be distinguishable from a complete one."""
    mock_post.return_value = _resp(200, "cut off mid-sen", finish_reason="length")
    out = call_model("http://fake", "key", "model", [{"role": "user", "content": "hi"}])
    assert out["finish_reason"] == "length"


@patch("runner.requests.post")
def test_call_model_sends_requested_max_tokens(mock_post):
    mock_post.return_value = _resp(200, "hi")
    call_model("http://fake", "key", "model", [{"role": "user", "content": "hi"}],
               max_tokens=1200)
    assert mock_post.call_args.kwargs["json"]["max_tokens"] == 1200


@patch("runner.requests.post")
def test_call_model_none_content_becomes_empty_string(mock_post):
    mock_post.return_value = _resp(200, None)
    out = call_model("http://fake", "key", "model", [{"role": "user", "content": "hi"}])
    assert out["content"] == ""


@patch("runner.time.sleep", return_value=None)
@patch("runner.call_model")
def test_run_one_builds_growing_message_history(mock_call_model, _sleep):
    replies = iter(["reply1", "reply2", "reply3"])
    mock_call_model.side_effect = lambda *a, **kw: {
        "content": next(replies), "finish_reason": "stop",
        "completion_tokens": 5, "total_tokens": 10,
    }

    task = {"id": "advisory_001", "turns": ["turn a", "turn b", "turn c"]}
    result = run_one(task, "groq", {"base_url": "http://x"}, "model-x", "key", 0)

    assert result["multiturn"] is True
    assert len(result["exchanges"]) == 3
    assert [e["assistant"] for e in result["exchanges"]] == ["reply1", "reply2", "reply3"]

    third_call_messages = mock_call_model.call_args_list[2].args[3]
    user_turns = [m["content"] for m in third_call_messages if m["role"] == "user"]
    assert user_turns == ["turn a", "turn b", "turn c"]


@patch("runner.time.sleep", return_value=None)
@patch("runner.call_model")
def test_run_one_single_turn_task_uses_prompt(mock_call_model, _sleep):
    mock_call_model.return_value = {
        "content": "the answer", "finish_reason": "stop",
        "completion_tokens": 5, "total_tokens": 10,
    }
    task = {"id": "research_001", "prompt": "what is x"}
    result = run_one(task, "groq", {"base_url": "http://x"}, "model-x", "key", 0)

    assert result["multiturn"] is False
    assert len(result["exchanges"]) == 1
    assert result["exchanges"][0]["user"] == "what is x"
    assert result["exchanges"][0]["assistant"] == "the answer"


@patch("runner.time.sleep", return_value=None)
@patch("runner.call_model")
def test_run_one_records_truncation_per_turn_and_in_total(mock_call_model, _sleep):
    """The bug this guards: a 250-token cap silently truncated 82% of replies
    and nothing in the saved transcript recorded it."""
    reasons = iter(["stop", "length", "length"])
    mock_call_model.side_effect = lambda *a, **kw: {
        "content": "text", "finish_reason": next(reasons),
        "completion_tokens": 5, "total_tokens": 10,
    }
    task = {"id": "advisory_001", "turns": ["a", "b", "c"]}
    result = run_one(task, "groq", {"base_url": "http://x"}, "model-x", "key", 0)

    assert [e["truncated"] for e in result["exchanges"]] == [False, True, True]
    assert result["n_truncated"] == 2
    assert result["max_tokens"] == 1200


def test_estimate_tokens_counts_history_plus_output_cap():
    messages = [{"role": "user", "content": "x" * 400}]
    assert estimate_tokens(messages, 1200) == 100 + 1200


def test_estimate_tokens_tolerates_none_content():
    messages = [{"role": "assistant", "content": None}, {"role": "user", "content": "abcd"}]
    assert estimate_tokens(messages, 0) == 1


@patch("runner.time.sleep", return_value=None)
def test_token_budget_allows_calls_under_the_cap(mock_sleep):
    b = TokenBudget(tpm=8000)
    b.reserve(3000)
    b.reserve(3000)
    assert mock_sleep.call_count == 0


@patch("runner.time.sleep", return_value=None)
def test_token_budget_waits_when_window_is_full(mock_sleep):
    b = TokenBudget(tpm=8000)
    b.reserve(5000)
    # window is pruned by wall time, which the mocked sleep doesn't advance, so
    # force the next prune to drop the old event and let the loop exit
    def _advance(_):
        b.events = []
    mock_sleep.side_effect = _advance
    b.reserve(5000)
    assert mock_sleep.call_count == 1


@patch("runner.time.sleep", return_value=None)
def test_token_budget_never_deadlocks_on_an_oversized_call(mock_sleep):
    """A single call bigger than the whole cap must go through, not hang."""
    b = TokenBudget(tpm=1000)
    b.reserve(50_000)
    assert mock_sleep.call_count == 0


@patch("runner.time.sleep", return_value=None)
def test_token_budget_settles_estimate_against_reported_usage(_sleep):
    b = TokenBudget(tpm=8000)
    b.reserve(5000)
    b.settle(120)
    assert b.events[-1][1] == 120


@patch("runner.time.sleep", return_value=None)
def test_token_budget_settle_ignores_missing_usage(_sleep):
    b = TokenBudget(tpm=8000)
    b.reserve(5000)
    b.settle(None)
    assert b.events[-1][1] == 5000


def test_load_system_prompt_reads_task_file_key(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("system: be brief\ntasks:\n  - id: a\n    prompt: hi\n", encoding="utf-8")
    from runner import load_system_prompt
    assert load_system_prompt(str(p)) == "be brief"


def test_load_system_prompt_absent_is_none(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("tasks:\n  - id: a\n    prompt: hi\n", encoding="utf-8")
    from runner import load_system_prompt
    assert load_system_prompt(str(p)) is None


@patch("runner.time.sleep", return_value=None)
@patch("runner.call_model")
def test_run_one_prepends_system_message_once(mock_call_model, _sleep):
    """The system prompt must lead the history and not repeat per turn."""
    mock_call_model.return_value = {
        "content": "r", "finish_reason": "stop",
        "completion_tokens": 5, "total_tokens": 10,
    }
    task = {"id": "advisory_001", "turns": ["a", "b", "c"]}
    result = run_one(task, "groq", {"base_url": "http://x"}, "m", "k", 0,
                     system="be brief")

    last_messages = mock_call_model.call_args_list[-1].args[3]
    assert last_messages[0] == {"role": "system", "content": "be brief"}
    assert sum(1 for m in last_messages if m["role"] == "system") == 1
    assert result["system"] == "be brief"


@patch("runner.time.sleep", return_value=None)
@patch("runner.call_model")
def test_run_one_without_system_sends_no_system_message(mock_call_model, _sleep):
    mock_call_model.return_value = {
        "content": "r", "finish_reason": "stop",
        "completion_tokens": 5, "total_tokens": 10,
    }
    task = {"id": "research_001", "prompt": "q"}
    result = run_one(task, "groq", {"base_url": "http://x"}, "m", "k", 0)

    sent = mock_call_model.call_args_list[-1].args[3]
    assert all(m["role"] != "system" for m in sent)
    assert result["system"] is None


def test_parse_duration_reads_groq_compact_durations():
    assert parse_duration("577ms") == 0.577
    assert parse_duration("7.66s") == 7.66
    assert parse_duration("1m26.4s") == 86.4


def test_parse_duration_reads_bare_seconds_from_retry_after():
    assert parse_duration("30") == 30.0


def test_parse_duration_returns_none_when_absent_or_unparseable():
    assert parse_duration(None) is None
    assert parse_duration("") is None
    assert parse_duration("soon") is None


@patch("runner.time.sleep", return_value=None)
@patch("runner.requests.post")
def test_429_waits_for_the_reset_the_server_reports(mock_post, mock_sleep):
    """The bug this guards: exponential backoff alone topped out at 30s total
    across 4 attempts, shorter than the 60s TPM window it was waiting on, so a
    run that overshot the cap could never recover."""
    mock_post.side_effect = [
        _resp(429, headers={"x-ratelimit-reset-tokens": "58.5s"}),
        _resp(200, "recovered"),
    ]
    out = call_model("http://fake", "key", "model", [{"role": "user", "content": "hi"}])
    assert out["content"] == "recovered"
    # 58.5s reported + 1s margin, not the 2s first exponential step
    assert mock_sleep.call_args_list[0].args[0] == pytest.approx(59.5)


@patch("runner.time.sleep", return_value=None)
@patch("runner.requests.post")
def test_429_prefers_retry_after_over_the_reset_header(mock_post, mock_sleep):
    mock_post.side_effect = [
        _resp(429, headers={"retry-after": "12",
                            "x-ratelimit-reset-tokens": "58.5s"}),
        _resp(200, "ok"),
    ]
    call_model("http://fake", "key", "model", [{"role": "user", "content": "hi"}])
    assert mock_sleep.call_args_list[0].args[0] == pytest.approx(13.0)


@patch("runner.time.sleep", return_value=None)
@patch("runner.requests.post")
def test_429_without_headers_still_backs_off_exponentially(mock_post, mock_sleep):
    mock_post.side_effect = [_resp(429), _resp(429), _resp(200, "ok")]
    call_model("http://fake", "key", "model", [{"role": "user", "content": "hi"}])
    assert [c.args[0] for c in mock_sleep.call_args_list] == [2, 4]


@patch("runner.time.sleep", return_value=None)
@patch("runner.requests.post")
def test_server_wait_is_capped_so_a_bad_header_cannot_stall_a_run(mock_post, mock_sleep):
    mock_post.side_effect = [_resp(429, headers={"retry-after": "9999"}),
                             _resp(200, "ok")]
    call_model("http://fake", "key", "model", [{"role": "user", "content": "hi"}],
               max_backoff=120)
    assert mock_sleep.call_args_list[0].args[0] == 120


@patch("runner.time.sleep", return_value=None)
@patch("runner.requests.post")
def test_exhausted_retries_raise_with_model_and_status(mock_post, _sleep):
    mock_post.return_value = _resp(429)
    with pytest.raises(RuntimeError) as excinfo:
        call_model("http://fake", "key", "model-x",
                   [{"role": "user", "content": "hi"}], max_retries=3)
    assert "429" in str(excinfo.value)
    assert "model-x" in str(excinfo.value)
    assert mock_post.call_count == 3


@patch("runner.time.sleep", return_value=None)
@patch("runner.requests.post")
def test_5xx_ignores_rate_limit_headers_and_uses_backoff(mock_post, mock_sleep):
    """A 503 is not a quota problem; a stale reset header must not extend it."""
    mock_post.side_effect = [
        _resp(503, headers={"x-ratelimit-reset-tokens": "58.5s"}),
        _resp(200, "ok"),
    ]
    call_model("http://fake", "key", "model", [{"role": "user", "content": "hi"}])
    assert mock_sleep.call_args_list[0].args[0] == 2
