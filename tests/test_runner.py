import json

import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from runner import (DailyQuotaExceeded, TokenBudget, call_model, estimate_tokens,
                    has_healthy_transcript, is_daily_quota_error, parse_duration,
                    parse_retry_hint, run_one, safe_slug)


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

# --- a per-day quota is not a retryable 429 ---------------------------------

GROQ_TPD_BODY = (
    '{"error":{"message":"Rate limit reached for model `openai/gpt-oss-120b` in '
    'organization `org_x` service tier `on_demand` on tokens per day (TPD): '
    'Limit 200000, Used 196936, Requested 4200.","code":"rate_limit_exceeded"}}'
)

GROQ_TPM_BODY = (
    '{"error":{"message":"Rate limit reached for model `openai/gpt-oss-120b` in '
    'organization `org_x` service tier `on_demand` on tokens per minute (TPM): '
    'Limit 8000, Used 7900, Requested 400.","code":"rate_limit_exceeded"}}'
)


def _resp_429(body, headers=None):
    r = MagicMock()
    r.status_code = 429
    r.text = body
    r.headers = headers or {}
    return r


def test_is_daily_quota_error_true_for_a_tpd_body():
    assert is_daily_quota_error(_resp_429(GROQ_TPD_BODY)) is True


def test_is_daily_quota_error_false_for_a_tpm_body():
    """The two 429s look identical apart from this word, and the headers carry
    only the per-minute figures, so the body is the sole signal."""
    assert is_daily_quota_error(_resp_429(GROQ_TPM_BODY)) is False


def test_is_daily_quota_error_survives_an_unreadable_body():
    r = MagicMock()
    type(r).text = PropertyMock(side_effect=Exception("no body"))
    assert is_daily_quota_error(r) is False


@patch("runner.time.sleep", return_value=None)
@patch("runner.requests.post")
def test_call_model_raises_immediately_on_a_daily_quota(mock_post, mock_sleep):
    """The bug this guards: 6 attempts and ~12 minutes spent discovering that a
    day-long window had not cleared, then repeated for every remaining task."""
    mock_post.return_value = _resp_429(GROQ_TPD_BODY)
    with pytest.raises(DailyQuotaExceeded):
        call_model("http://fake", "key", "openai/gpt-oss-120b",
                   [{"role": "user", "content": "hi"}])
    assert mock_post.call_count == 1
    assert mock_sleep.call_count == 0


@patch("runner.time.sleep", return_value=None)
@patch("runner.requests.post")
def test_daily_quota_error_names_the_model(mock_post, _sleep):
    mock_post.return_value = _resp_429(GROQ_TPD_BODY)
    with pytest.raises(DailyQuotaExceeded) as exc:
        call_model("http://fake", "key", "openai/gpt-oss-120b",
                   [{"role": "user", "content": "hi"}])
    assert "openai/gpt-oss-120b" in str(exc.value)


@patch("runner.time.sleep", return_value=None)
@patch("runner.requests.post")
def test_call_model_still_retries_a_per_minute_429(mock_post, mock_sleep):
    """Fail-fast must not swallow the minute-scale case, which does recover."""
    mock_post.side_effect = [_resp_429(GROQ_TPM_BODY), _resp(200, "recovered")]
    out = call_model("http://fake", "key", "openai/gpt-oss-120b",
                     [{"role": "user", "content": "hi"}])
    assert out["content"] == "recovered"
    assert mock_post.call_count == 2
    assert mock_sleep.call_count == 1


# --- resuming an aborted run ------------------------------------------------

def test_has_healthy_transcript_true_for_a_real_run(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"task_id": "a", "exchanges": [{"user": "u", "assistant": "a"}]}),
                 encoding="utf-8")
    assert has_healthy_transcript(p) is True


def test_has_healthy_transcript_false_for_an_error_record(tmp_path):
    """The bug this guards: a quota-killed run still writes a file, so
    "exists" must not mean "done" or a resume skips the repairs."""
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"task_id": "a", "error": "429 ..."}), encoding="utf-8")
    assert has_healthy_transcript(p) is False


def test_has_healthy_transcript_false_for_empty_exchanges(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"task_id": "a", "exchanges": []}), encoding="utf-8")
    assert has_healthy_transcript(p) is False


def test_has_healthy_transcript_false_for_a_missing_file(tmp_path):
    assert has_healthy_transcript(tmp_path / "nope.json") is False


def test_has_healthy_transcript_false_for_corrupt_json(tmp_path):
    p = tmp_path / "t.json"
    p.write_text("{not json", encoding="utf-8")
    assert has_healthy_transcript(p) is False


# --- a per-day quota is a rolling window, not a midnight reset --------------

GROQ_TPD_WITH_HINT = (
    '{"error":{"message":"Rate limit reached for model `openai/gpt-oss-120b` in '
    'organization `org_x` service tier `on_demand` on tokens per day (TPD): '
    'Limit 200000, Used 198674, Requested 6116. Please try again in 34m29.2s. '
    'Need more tokens? Upgrade to Dev Tier today at https://example."}}'
)


def test_parse_retry_hint_reads_the_quoted_delay():
    assert parse_retry_hint(GROQ_TPD_WITH_HINT) == pytest.approx(2069.2)


def test_parse_retry_hint_ignores_the_bare_numbers_in_the_same_message():
    """"Limit 200000, Used 198674" sits in the same string; a looser duration
    scan would read one of those as a wait."""
    hint = parse_retry_hint(GROQ_TPD_WITH_HINT)
    assert hint < 3600


def test_parse_retry_hint_returns_none_without_the_phrase():
    assert parse_retry_hint(GROQ_TPD_BODY) is None
    assert parse_retry_hint("") is None
    assert parse_retry_hint(None) is None


@patch("runner.time.sleep", return_value=None)
@patch("runner.requests.post")
def test_daily_quota_is_waited_out_when_the_delay_is_short(mock_post, mock_sleep):
    """Measured 2026-08-25: a 7000-token request succeeded seconds after a
    3000-token one was rejected, because headroom ages back in. Aborting on
    the first per-day 429 threw away a run that would have completed."""
    mock_post.side_effect = [_resp_429(GROQ_TPD_WITH_HINT), _resp(200, "made it")]
    out = call_model("http://fake", "key", "m", [{"role": "user", "content": "hi"}])
    assert out["content"] == "made it"
    assert mock_sleep.call_args_list[0].args[0] == pytest.approx(2074.2)


@patch("runner.time.sleep", return_value=None)
@patch("runner.requests.post")
def test_daily_quota_gives_up_when_the_delay_is_too_long(mock_post, mock_sleep):
    """A genuinely exhausted day must not hold the run open for hours."""
    mock_post.return_value = _resp_429(GROQ_TPD_WITH_HINT)
    with pytest.raises(DailyQuotaExceeded):
        call_model("http://fake", "key", "m", [{"role": "user", "content": "hi"}],
                   max_quota_wait=600)
    assert mock_post.call_count == 1
    assert mock_sleep.call_count == 0


@patch("runner.time.sleep", return_value=None)
@patch("runner.requests.post")
def test_daily_quota_waits_are_bounded(mock_post, mock_sleep):
    """Otherwise a saturated window could loop until max_retries ran out."""
    mock_post.return_value = _resp_429(GROQ_TPD_WITH_HINT)
    with pytest.raises(DailyQuotaExceeded):
        call_model("http://fake", "key", "m", [{"role": "user", "content": "hi"}],
                   max_quota_waits=2)
    assert mock_post.call_count == 3  # two waits, then the giving-up attempt


@patch("runner.time.sleep", return_value=None)
@patch("runner.requests.post")
def test_daily_quota_without_any_hint_still_aborts_at_once(mock_post, mock_sleep):
    """No quoted delay means no evidence the window will clear soon."""
    mock_post.return_value = _resp_429(GROQ_TPD_BODY)
    with pytest.raises(DailyQuotaExceeded):
        call_model("http://fake", "key", "m", [{"role": "user", "content": "hi"}])
    assert mock_post.call_count == 1
    assert mock_sleep.call_count == 0


@patch("runner.time.sleep", return_value=None)
@patch("runner.requests.post")
def test_retry_after_header_wins_over_the_body_hint(mock_post, mock_sleep):
    mock_post.side_effect = [
        _resp_429(GROQ_TPD_WITH_HINT, {"retry-after": "30"}),
        _resp(200, "ok"),
    ]
    call_model("http://fake", "key", "m", [{"role": "user", "content": "hi"}])
    assert mock_sleep.call_args_list[0].args[0] == pytest.approx(35.0)


@patch("runner.time.sleep", return_value=None)
@patch("runner.requests.post")
def test_daily_quota_message_reports_how_long_the_day_is_gone(mock_post, _sleep):
    mock_post.return_value = _resp_429(GROQ_TPD_WITH_HINT)
    with pytest.raises(DailyQuotaExceeded) as exc:
        call_model("http://fake", "key", "m", [{"role": "user", "content": "hi"}],
                   max_quota_wait=600)
    assert "34m" in str(exc.value)


GOOGLE_RPD_BODY = (
    '[{"error":{"code":429,"message":"You exceeded your current quota. '
    '* Quota exceeded for metric: generativelanguage.googleapis.com/'
    'generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash. '
    'Please retry in 40.8s.","status":"RESOURCE_EXHAUSTED","details":[{"@type":'
    '"type.googleapis.com/google.rpc.QuotaFailure","violations":[{"quotaId":'
    '"GenerateRequestsPerDayPerProjectPerModel-FreeTier","quotaValue":"20"}]},'
    '{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"40s"}]}}]'
)


def test_google_per_day_quota_is_recognised_despite_having_no_space():
    """Google writes PerDay with no space inside a quotaId. An earlier version
    of this check looked for "per day" and missed it, so a 20-requests-per-day
    cap was retried six times on the per-minute schedule."""
    assert is_daily_quota_error(_resp_429(GOOGLE_RPD_BODY)) is True


def test_a_plain_per_minute_body_is_still_not_a_daily_quota():
    assert is_daily_quota_error(_resp_429(GROQ_TPM_BODY)) is False
