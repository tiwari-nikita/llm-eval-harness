import json
from unittest.mock import MagicMock

import pytest

from catalogue import (API_PROVIDERS, LOCAL_RUNTIMES, PRODUCT_TIERS,
                       cmd_providers_diff, free_huggingface, free_openrouter,
                       free_sambanova, free_unknown, probe_api_provider,
                       probe_local_runtime, refresh, summarise)


def _resp(status=200, payload=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload if payload is not None else {}
    return r


def _getter(response):
    return lambda url, headers=None, timeout=None: response


# --- free-ness rules --------------------------------------------------------
# These decide what the whole catalogue claims, so they are tested against the
# real response shapes rather than invented ones.

def test_openrouter_zero_price_is_free():
    assert free_openrouter({"pricing": {"prompt": "0", "completion": "0"}}) is True


def test_openrouter_nonzero_price_is_paid():
    assert free_openrouter({"pricing": {"prompt": "0.0000004",
                                        "completion": "0.0000016"}}) is False


def test_openrouter_free_input_but_paid_output_is_not_free():
    """A model billed only on completion still costs money."""
    assert free_openrouter({"pricing": {"prompt": "0",
                                        "completion": "0.000002"}}) is False


def test_openrouter_missing_pricing_is_unknown_not_paid():
    """None and False must stay distinct: 'we don't know' and 'it costs money'
    lead to different next actions."""
    assert free_openrouter({"id": "x"}) is None


def test_openrouter_unparseable_price_is_unknown():
    assert free_openrouter({"pricing": {"prompt": "n/a"}}) is None


def test_sambanova_uses_the_same_price_shape():
    assert free_sambanova({"pricing": {"prompt": "0.00000300",
                                       "completion": "0.00000450"}}) is False


def test_huggingface_free_if_any_provider_is_free():
    entry = {"providers": [{"is_free": False}, {"is_free": True}]}
    assert free_huggingface(entry) is True


def test_huggingface_paid_when_every_provider_is_paid():
    """Verified against the live endpoint on 2026-08-25: all 132 models came
    back is_free=false on every provider. 0-free is the correct answer there,
    not a broken rule."""
    entry = {"providers": [{"is_free": False}, {"is_free": False}]}
    assert free_huggingface(entry) is False


def test_huggingface_no_providers_is_unknown():
    assert free_huggingface({"providers": []}) is None


def test_huggingface_non_boolean_flags_are_unknown():
    assert free_huggingface({"providers": [{"is_free": "yes"}]}) is None


def test_free_unknown_always_returns_none():
    assert free_unknown({"pricing": {"prompt": "0"}}) is None


# --- probing ----------------------------------------------------------------

def test_probe_classifies_each_model_by_provenance():
    payload = {"data": [
        {"id": "free/one", "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "paid/two", "pricing": {"prompt": "0.001", "completion": "0.002"}},
        {"id": "silent/three"},
    ]}
    out = probe_api_provider("openrouter", API_PROVIDERS["openrouter"],
                             getter=_getter(_resp(200, payload)))
    assert out["status"] == "live"
    by_id = {m["id"]: m for m in out["models"]}
    assert by_id["free/one"]["provenance"] == "verified_free"
    assert by_id["paid/two"]["provenance"] == "verified_paid"
    assert by_id["silent/three"]["provenance"] == "verified_listed"


def test_probe_reports_key_required_rather_than_empty():
    """A 401 must not look like 'this provider has no free models'."""
    out = probe_api_provider("groq", API_PROVIDERS["groq"],
                             getter=_getter(_resp(401)))
    assert out["status"] == "key_required"
    assert "GROQ_API_KEY" in out["error"]
    assert out["models"] == []


def test_probe_reports_unreachable_on_server_error():
    out = probe_api_provider("github", API_PROVIDERS["github"],
                             getter=_getter(_resp(410)))
    assert out["status"] == "unreachable"
    assert "410" in out["error"]


def test_probe_survives_a_network_failure():
    def boom(url, headers=None, timeout=None):
        raise ConnectionError("dns")
    out = probe_api_provider("openrouter", API_PROVIDERS["openrouter"], getter=boom)
    assert out["status"] == "unreachable"
    assert "ConnectionError" in out["error"]


def test_probe_survives_an_unparseable_body():
    r = MagicMock()
    r.status_code = 200
    r.json.side_effect = ValueError("not json")
    out = probe_api_provider("openrouter", API_PROVIDERS["openrouter"],
                             getter=_getter(r))
    assert out["status"] == "unreachable"


def test_probe_skips_entries_without_an_id():
    payload = {"data": [{"pricing": {"prompt": "0"}}, {"id": "ok"}]}
    out = probe_api_provider("openrouter", API_PROVIDERS["openrouter"],
                             getter=_getter(_resp(200, payload)))
    assert [m["id"] for m in out["models"]] == ["ok"]


def test_provider_without_a_models_url_is_documented_not_broken():
    out = probe_api_provider("cloudflare", API_PROVIDERS["cloudflare"],
                             getter=_getter(_resp(200, {})))
    assert out["status"] == "documented"
    assert "no global model-list endpoint" in out["error"]


def test_local_runtime_reports_running_with_models():
    payload = {"models": [{"name": "llama3:8b"}, {"name": "qwen:7b"}]}
    out = probe_local_runtime("ollama", LOCAL_RUNTIMES["ollama"],
                              getter=_getter(_resp(200, payload)))
    assert out["status"] == "running"
    assert [m["id"] for m in out["models"]] == ["llama3:8b", "qwen:7b"]
    assert all(m["free"] is True for m in out["models"])


def test_local_runtime_not_running_is_not_an_error():
    def refused(url, headers=None, timeout=None):
        raise ConnectionError("refused")
    out = probe_local_runtime("ollama", LOCAL_RUNTIMES["ollama"], getter=refused)
    assert out["status"] == "not_running"
    assert out["models"] == []


# --- assembly ---------------------------------------------------------------

def test_refresh_covers_every_registered_provider():
    out = refresh(env={}, getter=_getter(_resp(200, {"data": []})))
    names = {p["provider"] for p in out["providers"]}
    assert names == (set(API_PROVIDERS) | set(LOCAL_RUNTIMES) | set(PRODUCT_TIERS))


def test_refresh_marks_product_tiers_as_not_api_callable():
    out = refresh(env={}, getter=_getter(_resp(200, {"data": []})))
    products = [p for p in out["providers"] if p["kind"] == "product"]
    assert products
    assert all(p["openai_compatible"] is False for p in products)


def test_refresh_records_a_generation_timestamp():
    out = refresh(env={}, getter=_getter(_resp(200, {"data": []})))
    assert out["generated"].endswith("+00:00")


def test_summarise_separates_free_from_unknown():
    cat = {"providers": [{
        "provider": "x", "kind": "api", "status": "live", "models": [
            {"id": "a", "free": True}, {"id": "b", "free": False},
            {"id": "c", "free": None},
        ], "error": None,
    }]}
    row = summarise(cat)[0]
    assert (row["models"], row["free"], row["unknown"]) == (3, 1, 1)


# --- the check this module exists for --------------------------------------

def _write(tmp_path, catalogue, providers):
    c = tmp_path / "catalogue.json"
    c.write_text(json.dumps(catalogue), encoding="utf-8")
    p = tmp_path / "providers.yaml"
    p.write_text(providers, encoding="utf-8")
    return c, p


def test_providers_diff_flags_a_model_that_left_the_free_list(tmp_path, capsys):
    """The nvidia/nemotron incident: still listed, no longer free."""
    cat = {"providers": [{
        "provider": "openrouter", "kind": "api", "status": "live",
        "models": [{"id": "a:free", "free": True},
                   {"id": "b:free", "free": False}],
        "error": None,
    }]}
    c, p = _write(tmp_path, cat, "openrouter:\n  models:\n    - a:free\n    - b:free\n")
    args = MagicMock(catalogue=str(c), providers_file=str(p))
    assert cmd_providers_diff(args) == 1
    assert "NO LONGER FREE" in capsys.readouterr().out


def test_providers_diff_flags_a_model_that_vanished(tmp_path, capsys):
    """The llama-3.1-8b-instant incident: retired, 404s on use."""
    cat = {"providers": [{
        "provider": "groq", "kind": "api", "status": "live",
        "models": [{"id": "still-here", "free": None}], "error": None,
    }]}
    c, p = _write(tmp_path, cat, "groq:\n  models:\n    - still-here\n    - retired\n")
    args = MagicMock(catalogue=str(c), providers_file=str(p))
    assert cmd_providers_diff(args) == 1
    assert "GONE" in capsys.readouterr().out


def test_providers_diff_does_not_invent_problems_when_it_cannot_verify(tmp_path, capsys):
    """A key-gated provider must not be reported as stale just because this
    tool could not look."""
    cat = {"providers": [{
        "provider": "google", "kind": "api", "status": "key_required",
        "models": [], "error": "HTTP 403",
    }]}
    c, p = _write(tmp_path, cat, "google:\n  models:\n    - gemini-x\n")
    args = MagicMock(catalogue=str(c), providers_file=str(p))
    assert cmd_providers_diff(args) == 0
    assert "cannot verify" in capsys.readouterr().out


def test_providers_diff_passes_a_healthy_config(tmp_path):
    cat = {"providers": [{
        "provider": "openrouter", "kind": "api", "status": "live",
        "models": [{"id": "a:free", "free": True}], "error": None,
    }]}
    c, p = _write(tmp_path, cat, "openrouter:\n  models:\n    - a:free\n")
    args = MagicMock(catalogue=str(c), providers_file=str(p))
    assert cmd_providers_diff(args) == 0
