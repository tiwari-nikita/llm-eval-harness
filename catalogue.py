#!/usr/bin/env python3
"""
A live catalogue of free model access: hosted APIs, product tiers, and local runtimes.

Design principle, learned the hard way three times in this project: **hardcode
providers, discover models**. Provider facts (endpoint URL, auth style, signup
page) are stable and safe to write down. Model IDs and prices are not -- Groq
retired `llama-3.1-8b-instant` mid-project, and `nvidia/nemotron-3-nano-30b-a3b:free`
fell off OpenRouter's free list inside 48 hours. So no model ID in this file is
written by hand; every one is fetched, and anything that cannot be fetched is
reported as unknown rather than filled in from memory.

Every entry carries a provenance, and the distinction is the whole point:

    verified_free    an endpoint was queried just now and said price == 0
    verified_listed  the model list is live, but the endpoint exposes no
                     pricing, so "is it free" is not machine-checkable here
    key_required     a live list exists but needs an API key we do not have
    documented       no machine-checkable source; provider-level facts only,
                     recorded without invented quota numbers
    running / not_running   for local runtimes, probed on localhost

Usage:
    python catalogue.py refresh                 # probe everything reachable
    python catalogue.py list --free-only
    python catalogue.py list --eval-usable --provider openrouter
    python catalogue.py export -o FREE_MODELS.md
    python catalogue.py providers-diff          # vs providers.yaml
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
CATALOGUE_PATH = HERE / "catalogue.json"
DEFAULT_TIMEOUT = 25


# --------------------------------------------------------------------------
# free-ness rules
#
# One per response shape. Each returns True (free), False (paid), or None
# (this endpoint does not say). None is a real answer and must not be
# collapsed into False -- "we don't know" and "it costs money" lead to very
# different next actions.
# --------------------------------------------------------------------------

def _price_is_zero(value):
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return None


def free_openrouter(entry):
    pricing = entry.get("pricing") or {}
    if "prompt" not in pricing:
        return None
    prompt = _price_is_zero(pricing.get("prompt"))
    completion = _price_is_zero(pricing.get("completion"))
    if prompt is None:
        return None
    # a model billed only on output is still not free
    return bool(prompt and (completion is not False))


def free_sambanova(entry):
    return free_openrouter(entry)


def free_huggingface(entry):
    """HF fans a model out across several inference providers; free if any
    one of them serves it free."""
    providers = entry.get("providers")
    if not isinstance(providers, list) or not providers:
        return None
    flags = [p.get("is_free") for p in providers if isinstance(p, dict)]
    flags = [f for f in flags if isinstance(f, bool)]
    if not flags:
        return None
    return any(flags)


def free_unknown(entry):
    return None


# --------------------------------------------------------------------------
# provider registry
# --------------------------------------------------------------------------

API_PROVIDERS = {
    "openrouter": {
        "org": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "models_url": "https://openrouter.ai/api/v1/models",
        "public_list": True,
        "api_key_env": "OPENROUTER_API_KEY",
        "signup": "https://openrouter.ai/keys",
        "free_rule": free_openrouter,
        "openai_compatible": True,
        "notes": "Aggregator. Publishes per-model pricing, so free-tier "
                 "membership is verifiable per model rather than per account. "
                 "The `:free` suffix is a naming convention, not the source of "
                 "truth -- price is.",
    },
    "huggingface": {
        "org": "Hugging Face",
        "base_url": "https://router.huggingface.co/v1",
        "models_url": "https://router.huggingface.co/v1/models",
        "public_list": True,
        "api_key_env": "HF_TOKEN",
        "signup": "https://huggingface.co/settings/tokens",
        "free_rule": free_huggingface,
        "openai_compatible": True,
        "notes": "Router fans each model out to several inference providers, "
                 "each with its own is_free flag. Probed 2026-08-25: all 132 "
                 "models reported is_free=false on every provider, so the "
                 "router is a paid surface. HF's free allowance is "
                 "account-credit-based on a different surface, which this "
                 "endpoint cannot see.",
    },
    "sambanova": {
        "org": "SambaNova",
        "base_url": "https://api.sambanova.ai/v1",
        "models_url": "https://api.sambanova.ai/v1/models",
        "public_list": True,
        "api_key_env": "SAMBANOVA_API_KEY",
        "signup": "https://cloud.sambanova.ai",
        "free_rule": free_sambanova,
        "openai_compatible": True,
        "notes": "Small catalogue, publishes per-model pricing.",
    },
    "nvidia": {
        "org": "NVIDIA",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "models_url": "https://integrate.api.nvidia.com/v1/models",
        "public_list": True,
        "api_key_env": "NVIDIA_API_KEY",
        "signup": "https://build.nvidia.com",
        "free_rule": free_unknown,
        "openai_compatible": True,
        "notes": "Model list is public but carries no pricing field, so this "
                 "tool cannot verify free-ness. NIM access is credit-based at "
                 "the account level -- check the account page, not the model.",
    },
    "groq": {
        "org": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "models_url": "https://api.groq.com/openai/v1/models",
        "public_list": False,
        "api_key_env": "GROQ_API_KEY",
        "signup": "https://console.groq.com/keys",
        "free_rule": free_unknown,
        "openai_compatible": True,
        "notes": "Free tier is an account-level rate limit (TPM and TPD), not "
                 "a per-model price, so no endpoint can answer 'is this model "
                 "free'. Limits arrive in x-ratelimit-* response headers; the "
                 "per-day cap only appears in a 429 body.",
    },
    "google": {
        "org": "Google AI Studio",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "models_url": "https://generativelanguage.googleapis.com/v1beta/models",
        "public_list": False,
        "api_key_env": "GOOGLE_API_KEY",
        "signup": "https://aistudio.google.com/apikey",
        "free_rule": free_unknown,
        "openai_compatible": True,
        "notes": "Free tier is account-level. Different model family from "
                 "Groq's gpt-oss, which is what makes it the useful second key "
                 "for this eval: a grader that is not related to either model "
                 "it grades.",
    },
    "cerebras": {
        "org": "Cerebras",
        "base_url": "https://api.cerebras.ai/v1",
        "models_url": "https://api.cerebras.ai/v1/models",
        "public_list": False,
        "api_key_env": "CEREBRAS_API_KEY",
        "signup": "https://cloud.cerebras.ai",
        "free_rule": free_unknown,
        "openai_compatible": True,
        "notes": "Free tier is account-level.",
    },
    "mistral": {
        "org": "Mistral",
        "base_url": "https://api.mistral.ai/v1",
        "models_url": "https://api.mistral.ai/v1/models",
        "public_list": False,
        "api_key_env": "MISTRAL_API_KEY",
        "signup": "https://console.mistral.ai",
        "free_rule": free_unknown,
        "openai_compatible": True,
        "notes": "Free experiment tier is account-level.",
    },
    "together": {
        "org": "Together AI",
        "base_url": "https://api.together.xyz/v1",
        "models_url": "https://api.together.xyz/v1/models",
        "public_list": False,
        "api_key_env": "TOGETHER_API_KEY",
        "signup": "https://api.together.ai/settings/api-keys",
        "free_rule": free_unknown,
        "openai_compatible": True,
        "notes": "Some models carry a free variant; needs a key to enumerate.",
    },
    "github": {
        "org": "GitHub Models",
        "base_url": "https://models.github.ai/inference",
        "models_url": "https://models.github.ai/catalog/models",
        "public_list": False,
        "api_key_env": "GITHUB_TOKEN",
        "signup": "https://github.com/marketplace/models",
        "free_rule": free_unknown,
        "openai_compatible": True,
        "notes": "Catalogue endpoint returned HTTP 410 unauthenticated when "
                 "last probed; treat the URL as needing a token or as moved.",
    },
    "cloudflare": {
        "org": "Cloudflare Workers AI",
        "base_url": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        "models_url": None,
        "public_list": False,
        "api_key_env": "CLOUDFLARE_API_TOKEN",
        "signup": "https://dash.cloudflare.com/profile/api-tokens",
        "free_rule": free_unknown,
        "openai_compatible": True,
        "notes": "Endpoint is account-scoped, so there is no global model URL "
                 "to probe without an account id. Daily free allowance.",
    },
}

# Free access that exists but cannot be driven by runner.py. Recorded because
# it was asked for, deliberately without quota numbers: those change often and
# anything written here would be an unverifiable claim from memory. Only the
# stable facts (vendor, what it is, where to check) are stored; the URL is
# reachability-checked on refresh so at least that much is live.
PRODUCT_TIERS = {
    "claude_code": {"org": "Anthropic", "what": "Agentic coding CLI/IDE/web",
                    "url": "https://claude.com/product/claude-code"},
    "claude_web": {"org": "Anthropic", "what": "Claude chat, free tier",
                   "url": "https://claude.ai"},
    "chatgpt_free": {"org": "OpenAI", "what": "ChatGPT, free tier",
                     "url": "https://chatgpt.com"},
    "gemini_web": {"org": "Google", "what": "Gemini chat, free tier",
                   "url": "https://gemini.google.com"},
    "gemini_cli": {"org": "Google", "what": "Open-source terminal agent",
                   "url": "https://github.com/google-gemini/gemini-cli"},
    "github_copilot": {"org": "GitHub", "what": "Coding assistant, free tier",
                       "url": "https://github.com/features/copilot"},
    "cursor": {"org": "Anysphere", "what": "AI editor, free tier",
               "url": "https://cursor.com"},
    "windsurf": {"org": "Windsurf", "what": "AI editor, free tier",
                 "url": "https://windsurf.com"},
    "zed": {"org": "Zed Industries", "what": "Editor with agentic AI",
            "url": "https://zed.dev"},
    "cline": {"org": "Cline", "what": "Open-source coding agent extension",
              "url": "https://cline.bot"},
    "aider": {"org": "Aider", "what": "Open-source terminal coding agent",
              "url": "https://aider.chat"},
    "opencode": {"org": "OpenCode", "what": "Open-source terminal coding agent",
                 "url": "https://opencode.ai"},
    "grok_web": {"org": "xAI", "what": "Grok chat, free tier",
                 "url": "https://grok.com"},
    "deepseek_web": {"org": "DeepSeek", "what": "DeepSeek chat, free",
                     "url": "https://chat.deepseek.com"},
    "qwen_chat": {"org": "Alibaba", "what": "Qwen chat, free",
                  "url": "https://chat.qwen.ai"},
    "mistral_chat": {"org": "Mistral", "what": "Le Chat, free tier",
                     "url": "https://chat.mistral.ai"},
    "perplexity": {"org": "Perplexity", "what": "Answer engine, free tier",
                   "url": "https://perplexity.ai"},
    "kimi": {"org": "Moonshot AI", "what": "Kimi chat, free",
             "url": "https://kimi.com"},
    "zai_chat": {"org": "Z.ai", "what": "GLM chat, free tier",
                 "url": "https://chat.z.ai"},
}

# Local runtimes: free by construction, and unlike everything else here their
# availability is directly checkable -- either the port answers or it doesn't.
LOCAL_RUNTIMES = {
    "ollama": {"org": "Ollama", "probe": "http://localhost:11434/api/tags",
               "list_key": "models", "id_key": "name",
               "url": "https://ollama.com",
               "openai_base": "http://localhost:11434/v1"},
    "lm_studio": {"org": "LM Studio", "probe": "http://localhost:1234/v1/models",
                  "list_key": "data", "id_key": "id",
                  "url": "https://lmstudio.ai",
                  "openai_base": "http://localhost:1234/v1"},
    "llama_cpp": {"org": "llama.cpp", "probe": "http://localhost:8080/v1/models",
                  "list_key": "data", "id_key": "id",
                  "url": "https://github.com/ggml-org/llama.cpp",
                  "openai_base": "http://localhost:8080/v1"},
    "vllm": {"org": "vLLM", "probe": "http://localhost:8000/v1/models",
             "list_key": "data", "id_key": "id",
             "url": "https://github.com/vllm-project/vllm",
             "openai_base": "http://localhost:8000/v1"},
}


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------

def _get(url, headers=None, timeout=DEFAULT_TIMEOUT):
    return requests.get(url, headers=headers, timeout=timeout)


def _extract_list(payload):
    if isinstance(payload, dict):
        for key in ("data", "models", "body"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return []
    return payload if isinstance(payload, list) else []


def probe_api_provider(name, spec, api_key=None, getter=_get):
    """Fetch one provider's model list and classify each entry."""
    result = {
        "provider": name,
        "kind": "api",
        "org": spec["org"],
        "base_url": spec["base_url"],
        "api_key_env": spec["api_key_env"],
        "signup": spec["signup"],
        "openai_compatible": spec["openai_compatible"],
        "notes": spec["notes"],
        "models": [],
        "status": None,
        "error": None,
    }

    if not spec.get("models_url"):
        result["status"] = "documented"
        result["error"] = "no global model-list endpoint exists for this provider"
        return result

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        resp = getter(spec["models_url"], headers=headers)
    except Exception as e:
        result["status"] = "unreachable"
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    if resp.status_code in (401, 403):
        result["status"] = "key_required"
        result["error"] = f"HTTP {resp.status_code}; set {spec['api_key_env']}"
        return result
    if resp.status_code >= 400:
        result["status"] = "unreachable"
        result["error"] = f"HTTP {resp.status_code}"
        return result

    try:
        entries = _extract_list(resp.json())
    except ValueError as e:
        result["status"] = "unreachable"
        result["error"] = f"unparseable response: {e}"
        return result

    rule = spec["free_rule"]
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        is_free = rule(entry)
        result["models"].append({
            "id": entry["id"],
            "free": is_free,
            "provenance": "verified_free" if is_free
                          else "verified_paid" if is_free is False
                          else "verified_listed",
            "context_length": entry.get("context_length"),
        })
    result["status"] = "live"
    return result


def probe_local_runtime(name, spec, getter=_get):
    result = {
        "provider": name, "kind": "local", "org": spec["org"],
        "base_url": spec["openai_base"], "signup": spec["url"],
        "api_key_env": None, "openai_compatible": True,
        "notes": "Runs on your own hardware; free by construction.",
        "models": [], "status": None, "error": None,
    }
    try:
        resp = getter(spec["probe"], timeout=3)
    except Exception:
        result["status"] = "not_running"
        result["error"] = "nothing listening on the default port"
        return result
    if resp.status_code >= 400:
        result["status"] = "not_running"
        result["error"] = f"HTTP {resp.status_code}"
        return result
    try:
        payload = resp.json()
    except ValueError:
        result["status"] = "not_running"
        result["error"] = "unparseable response"
        return result
    for entry in _extract_list(payload):
        if isinstance(entry, dict) and entry.get(spec["id_key"]):
            result["models"].append({
                "id": entry[spec["id_key"]], "free": True,
                "provenance": "running", "context_length": None,
            })
    result["status"] = "running"
    return result


def probe_product_tier(name, spec, getter=_get):
    result = {
        "provider": name, "kind": "product", "org": spec["org"],
        "base_url": None, "signup": spec["url"], "api_key_env": None,
        "openai_compatible": False,
        "notes": spec["what"] + ". Free access exists but is not API-callable, "
                 "so runner.py cannot drive it. Quota deliberately not recorded "
                 "here -- it changes often and any number would be unverified.",
        "models": [], "status": "documented", "error": None,
    }
    try:
        resp = getter(spec["url"], timeout=10)
        result["url_reachable"] = resp.status_code < 400
    except Exception:
        result["url_reachable"] = None
    return result


def refresh(env=None, getter=_get, include_products=True, include_local=True):
    env = env if env is not None else __import__("os").environ
    providers = []
    for name, spec in API_PROVIDERS.items():
        key = env.get(spec["api_key_env"]) if spec.get("api_key_env") else None
        providers.append(probe_api_provider(name, spec, api_key=key, getter=getter))
    if include_local:
        for name, spec in LOCAL_RUNTIMES.items():
            providers.append(probe_local_runtime(name, spec, getter=getter))
    if include_products:
        for name, spec in PRODUCT_TIERS.items():
            providers.append(probe_product_tier(name, spec, getter=getter))

    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "provenance_note": (
            "verified_free/paid: an endpoint published a price. verified_listed: "
            "the model exists but the endpoint publishes no price, so free-ness "
            "is unknown here. key_required: a live list exists but needs a key. "
            "documented: provider-level facts only, no machine-checkable source. "
            "No model ID in this file was written by hand."
        ),
        "providers": providers,
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def summarise(cat):
    rows = []
    for p in cat["providers"]:
        free = sum(1 for m in p["models"] if m["free"] is True)
        unknown = sum(1 for m in p["models"] if m["free"] is None)
        rows.append({
            "provider": p["provider"], "kind": p["kind"], "status": p["status"],
            "models": len(p["models"]), "free": free, "unknown": unknown,
            "error": p.get("error"),
        })
    return rows


def load_catalogue(path=CATALOGUE_PATH):
    if not Path(path).exists():
        sys.exit(f"no catalogue at {path}; run `python catalogue.py refresh` first")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cmd_refresh(args):
    cat = refresh()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cat, f, ensure_ascii=False, indent=2)
    print(f"catalogue written to {args.out}  ({cat['generated']})\n")
    hdr = f"{'provider':14s} {'kind':8s} {'status':13s} {'models':>7s} {'free':>6s} {'?':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for r in summarise(cat):
        note = f"  {r['error']}" if r["error"] else ""
        print(f"{r['provider']:14s} {r['kind']:8s} {r['status'] or '':13s} "
              f"{r['models']:>7d} {r['free']:>6d} {r['unknown']:>6d}{note}")


def cmd_list(args):
    cat = load_catalogue(args.catalogue)
    shown = 0
    for p in cat["providers"]:
        if args.provider and p["provider"] != args.provider:
            continue
        if args.eval_usable and not p["openai_compatible"]:
            continue
        models = p["models"]
        if args.free_only:
            models = [m for m in models if m["free"] is True]
        if not models:
            continue
        print(f"\n## {p['provider']}  ({p['kind']}, {p['status']})")
        for m in sorted(models, key=lambda m: m["id"]):
            ctx = f"  ctx={m['context_length']}" if m.get("context_length") else ""
            print(f"  {m['id']:60s} {m['provenance']}{ctx}")
            shown += 1
    print(f"\n{shown} models")


def cmd_export(args):
    cat = load_catalogue(args.catalogue)
    lines = [
        "# Free model access — live catalogue",
        "",
        f"Generated {cat['generated']} by `catalogue.py`.",
        "",
        "> No model ID here was typed by hand. Everything under a *live* "
        "provider was fetched from that provider's own endpoint at the time "
        "above. Free-tier membership drifts within days, so regenerate rather "
        "than trust this file's age.",
        "",
        cat["provenance_note"],
        "",
        "## Summary",
        "",
        "| provider | kind | status | models | free | unknown |",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in summarise(cat):
        lines.append(f"| {r['provider']} | {r['kind']} | {r['status']} | "
                     f"{r['models']} | {r['free']} | {r['unknown']} |")

    for kind, title in (("api", "Hosted APIs"), ("local", "Local runtimes"),
                        ("product", "Product tiers (not API-callable)")):
        lines += ["", f"## {title}", ""]
        for p in cat["providers"]:
            if p["kind"] != kind:
                continue
            lines.append(f"### {p['provider']} — {p['org']}")
            lines.append("")
            if p.get("base_url"):
                lines.append(f"- endpoint: `{p['base_url']}`")
            if p.get("api_key_env"):
                lines.append(f"- key env: `{p['api_key_env']}`")
            lines.append(f"- signup: {p['signup']}")
            lines.append(f"- status: `{p['status']}`"
                         + (f" — {p['error']}" if p.get("error") else ""))
            lines.append(f"- {p['notes']}")
            free = [m for m in p["models"] if m["free"] is True]
            if free:
                lines += ["", f"<details><summary>{len(free)} free models</summary>", ""]
                lines += [f"- `{m['id']}`" for m in sorted(free, key=lambda m: m["id"])]
                lines += ["", "</details>"]
            elif p["models"]:
                lines.append(f"- {len(p['models'])} models listed, none "
                             f"machine-verifiable as free")
            lines.append("")

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out} ({len(lines)} lines)")


def cmd_providers_diff(args):
    """Check providers.yaml's model IDs against the live catalogue.

    This is the check that would have caught every stale-model incident this
    project has had.
    """
    cat = load_catalogue(args.catalogue)
    by_name = {p["provider"]: p for p in cat["providers"]}
    with open(args.providers_file, encoding="utf-8") as f:
        configured = yaml.safe_load(f) or {}

    problems = 0
    for pname, cfg in configured.items():
        live = by_name.get(pname)
        print(f"\n[{pname}]")
        if not live:
            print("  not in catalogue")
            continue
        if live["status"] != "live":
            print(f"  catalogue status {live['status']}"
                  + (f" ({live['error']})" if live.get("error") else "")
                  + " — cannot verify, leaving as configured")
            continue
        free_ids = {m["id"] for m in live["models"] if m["free"] is True}
        all_ids = {m["id"] for m in live["models"]}
        for model in cfg.get("models", []):
            if model not in all_ids:
                print(f"  {model}  GONE — not in the live model list")
                problems += 1
            elif free_ids and model not in free_ids:
                print(f"  {model}  NO LONGER FREE")
                problems += 1
            else:
                print(f"  {model}  ok")
    print(f"\n{problems} problem(s)")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("refresh", help="probe every reachable provider")
    r.add_argument("-o", "--out", default=str(CATALOGUE_PATH))
    r.set_defaults(func=cmd_refresh)

    l = sub.add_parser("list", help="list catalogued models")
    l.add_argument("--catalogue", default=str(CATALOGUE_PATH))
    l.add_argument("--free-only", action="store_true")
    l.add_argument("--eval-usable", action="store_true",
                   help="only OpenAI-compatible endpoints runner.py can drive")
    l.add_argument("--provider")
    l.set_defaults(func=cmd_list)

    e = sub.add_parser("export", help="write a readable Markdown catalogue")
    e.add_argument("--catalogue", default=str(CATALOGUE_PATH))
    e.add_argument("-o", "--out", default=str(HERE / "FREE_MODELS.md"))
    e.set_defaults(func=cmd_export)

    d = sub.add_parser("providers-diff",
                       help="check providers.yaml against the live catalogue")
    d.add_argument("--catalogue", default=str(CATALOGUE_PATH))
    d.add_argument("--providers-file", default=str(HERE / "providers.yaml"))
    d.set_defaults(func=cmd_providers_diff)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
