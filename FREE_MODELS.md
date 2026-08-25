# Free model access — live catalogue

Generated 2026-08-25T22:54:38.490612+00:00 by `catalogue.py`.

> No model ID here was typed by hand. Everything under a *live* provider was fetched from that provider's own endpoint at the time above. Free-tier membership drifts within days, so regenerate rather than trust this file's age.

verified_free/paid: an endpoint published a price. verified_listed: the model exists but the endpoint publishes no price, so free-ness is unknown here. key_required: a live list exists but needs a key. documented: provider-level facts only, no machine-checkable source. No model ID in this file was written by hand.

## Summary

| provider | kind | status | models | free | unknown |
|---|---|---|---:|---:|---:|
| openrouter | api | live | 417 | 21 | 0 |
| huggingface | api | live | 132 | 0 | 0 |
| sambanova | api | live | 7 | 0 | 0 |
| nvidia | api | live | 95 | 0 | 95 |
| groq | api | live | 13 | 0 | 13 |
| google | api | key_required | 0 | 0 | 0 |
| cerebras | api | key_required | 0 | 0 | 0 |
| mistral | api | key_required | 0 | 0 | 0 |
| together | api | key_required | 0 | 0 | 0 |
| github | api | unreachable | 0 | 0 | 0 |
| cloudflare | api | documented | 0 | 0 | 0 |
| ollama | local | not_running | 0 | 0 | 0 |
| lm_studio | local | not_running | 0 | 0 | 0 |
| llama_cpp | local | not_running | 0 | 0 | 0 |
| vllm | local | not_running | 0 | 0 | 0 |
| claude_code | product | documented | 0 | 0 | 0 |
| claude_web | product | documented | 0 | 0 | 0 |
| chatgpt_free | product | documented | 0 | 0 | 0 |
| gemini_web | product | documented | 0 | 0 | 0 |
| gemini_cli | product | documented | 0 | 0 | 0 |
| github_copilot | product | documented | 0 | 0 | 0 |
| cursor | product | documented | 0 | 0 | 0 |
| windsurf | product | documented | 0 | 0 | 0 |
| zed | product | documented | 0 | 0 | 0 |
| cline | product | documented | 0 | 0 | 0 |
| aider | product | documented | 0 | 0 | 0 |
| opencode | product | documented | 0 | 0 | 0 |
| grok_web | product | documented | 0 | 0 | 0 |
| deepseek_web | product | documented | 0 | 0 | 0 |
| qwen_chat | product | documented | 0 | 0 | 0 |
| mistral_chat | product | documented | 0 | 0 | 0 |
| perplexity | product | documented | 0 | 0 | 0 |
| kimi | product | documented | 0 | 0 | 0 |
| zai_chat | product | documented | 0 | 0 | 0 |

## Hosted APIs

### openrouter — OpenRouter

- endpoint: `https://openrouter.ai/api/v1`
- key env: `OPENROUTER_API_KEY`
- signup: https://openrouter.ai/keys
- status: `live`
- Aggregator. Publishes per-model pricing, so free-tier membership is verifiable per model rather than per account. The `:free` suffix is a naming convention, not the source of truth -- price is.

<details><summary>21 free models</summary>

- `cohere/north-mini-code:free`
- `dots-studio/dots-3-note-preview:free`
- `google/gemma-4-26b-a4b-it:free`
- `google/gemma-4-31b-it:free`
- `google/lyria-3-clip-preview`
- `google/lyria-3-pro-preview`
- `liquid/lfm-2.5-2.6b:free`
- `minimax/minimax-m2.7:free`
- `minimax/minimax-m3:free`
- `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`
- `nvidia/nemotron-3-super-120b-a12b:free`
- `nvidia/nemotron-3-ultra-550b-a55b:free`
- `nvidia/nemotron-3.5-content-safety:free`
- `nvidia/nemotron-3.5-lightning:free`
- `openrouter/free`
- `poolside/laguna-s-2.1:free`
- `poolside/laguna-xs-2.1:free`
- `stealth/ox-alpha`
- `thinkingmachines/inkling-small:free`
- `thinkingmachines/inkling:free`
- `z-ai/glm-5.2:free`

</details>

### huggingface — Hugging Face

- endpoint: `https://router.huggingface.co/v1`
- key env: `HF_TOKEN`
- signup: https://huggingface.co/settings/tokens
- status: `live`
- Router fans each model out to several inference providers, each with its own is_free flag.
- 132 models listed, none machine-verifiable as free

### sambanova — SambaNova

- endpoint: `https://api.sambanova.ai/v1`
- key env: `SAMBANOVA_API_KEY`
- signup: https://cloud.sambanova.ai
- status: `live`
- Small catalogue, publishes per-model pricing.
- 7 models listed, none machine-verifiable as free

### nvidia — NVIDIA

- endpoint: `https://integrate.api.nvidia.com/v1`
- key env: `NVIDIA_API_KEY`
- signup: https://build.nvidia.com
- status: `live`
- Model list is public but carries no pricing field, so this tool cannot verify free-ness. NIM access is credit-based at the account level -- check the account page, not the model.
- 95 models listed, none machine-verifiable as free

### groq — Groq

- endpoint: `https://api.groq.com/openai/v1`
- key env: `GROQ_API_KEY`
- signup: https://console.groq.com/keys
- status: `live`
- Free tier is an account-level rate limit (TPM and TPD), not a per-model price, so no endpoint can answer 'is this model free'. Limits arrive in x-ratelimit-* response headers; the per-day cap only appears in a 429 body.
- 13 models listed, none machine-verifiable as free

### google — Google AI Studio

- endpoint: `https://generativelanguage.googleapis.com/v1beta/openai`
- key env: `GOOGLE_API_KEY`
- signup: https://aistudio.google.com/apikey
- status: `key_required` — HTTP 403; set GOOGLE_API_KEY
- Free tier is account-level. Different model family from Groq's gpt-oss, which is what makes it the useful second key for this eval: a grader that is not related to either model it grades.

### cerebras — Cerebras

- endpoint: `https://api.cerebras.ai/v1`
- key env: `CEREBRAS_API_KEY`
- signup: https://cloud.cerebras.ai
- status: `key_required` — HTTP 403; set CEREBRAS_API_KEY
- Free tier is account-level.

### mistral — Mistral

- endpoint: `https://api.mistral.ai/v1`
- key env: `MISTRAL_API_KEY`
- signup: https://console.mistral.ai
- status: `key_required` — HTTP 401; set MISTRAL_API_KEY
- Free experiment tier is account-level.

### together — Together AI

- endpoint: `https://api.together.xyz/v1`
- key env: `TOGETHER_API_KEY`
- signup: https://api.together.ai/settings/api-keys
- status: `key_required` — HTTP 401; set TOGETHER_API_KEY
- Some models carry a free variant; needs a key to enumerate.

### github — GitHub Models

- endpoint: `https://models.github.ai/inference`
- key env: `GITHUB_TOKEN`
- signup: https://github.com/marketplace/models
- status: `unreachable` — HTTP 410
- Catalogue endpoint returned HTTP 410 unauthenticated when last probed; treat the URL as needing a token or as moved.

### cloudflare — Cloudflare Workers AI

- endpoint: `https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1`
- key env: `CLOUDFLARE_API_TOKEN`
- signup: https://dash.cloudflare.com/profile/api-tokens
- status: `documented` — no global model-list endpoint exists for this provider
- Endpoint is account-scoped, so there is no global model URL to probe without an account id. Daily free allowance.


## Local runtimes

### ollama — Ollama

- endpoint: `http://localhost:11434/v1`
- signup: https://ollama.com
- status: `not_running` — nothing listening on the default port
- Runs on your own hardware; free by construction.

### lm_studio — LM Studio

- endpoint: `http://localhost:1234/v1`
- signup: https://lmstudio.ai
- status: `not_running` — nothing listening on the default port
- Runs on your own hardware; free by construction.

### llama_cpp — llama.cpp

- endpoint: `http://localhost:8080/v1`
- signup: https://github.com/ggml-org/llama.cpp
- status: `not_running` — nothing listening on the default port
- Runs on your own hardware; free by construction.

### vllm — vLLM

- endpoint: `http://localhost:8000/v1`
- signup: https://github.com/vllm-project/vllm
- status: `not_running` — nothing listening on the default port
- Runs on your own hardware; free by construction.


## Product tiers (not API-callable)

### claude_code — Anthropic

- signup: https://claude.com/product/claude-code
- status: `documented`
- Agentic coding CLI/IDE/web. Free access exists but is not API-callable, so runner.py cannot drive it. Quota deliberately not recorded here -- it changes often and any number would be unverified.

### claude_web — Anthropic

- signup: https://claude.ai
- status: `documented`
- Claude chat, free tier. Free access exists but is not API-callable, so runner.py cannot drive it. Quota deliberately not recorded here -- it changes often and any number would be unverified.

### chatgpt_free — OpenAI

- signup: https://chatgpt.com
- status: `documented`
- ChatGPT, free tier. Free access exists but is not API-callable, so runner.py cannot drive it. Quota deliberately not recorded here -- it changes often and any number would be unverified.

### gemini_web — Google

- signup: https://gemini.google.com
- status: `documented`
- Gemini chat, free tier. Free access exists but is not API-callable, so runner.py cannot drive it. Quota deliberately not recorded here -- it changes often and any number would be unverified.

### gemini_cli — Google

- signup: https://github.com/google-gemini/gemini-cli
- status: `documented`
- Open-source terminal agent. Free access exists but is not API-callable, so runner.py cannot drive it. Quota deliberately not recorded here -- it changes often and any number would be unverified.

### github_copilot — GitHub

- signup: https://github.com/features/copilot
- status: `documented`
- Coding assistant, free tier. Free access exists but is not API-callable, so runner.py cannot drive it. Quota deliberately not recorded here -- it changes often and any number would be unverified.

### cursor — Anysphere

- signup: https://cursor.com
- status: `documented`
- AI editor, free tier. Free access exists but is not API-callable, so runner.py cannot drive it. Quota deliberately not recorded here -- it changes often and any number would be unverified.

### windsurf — Windsurf

- signup: https://windsurf.com
- status: `documented`
- AI editor, free tier. Free access exists but is not API-callable, so runner.py cannot drive it. Quota deliberately not recorded here -- it changes often and any number would be unverified.

### zed — Zed Industries

- signup: https://zed.dev
- status: `documented`
- Editor with agentic AI. Free access exists but is not API-callable, so runner.py cannot drive it. Quota deliberately not recorded here -- it changes often and any number would be unverified.

### cline — Cline

- signup: https://cline.bot
- status: `documented`
- Open-source coding agent extension. Free access exists but is not API-callable, so runner.py cannot drive it. Quota deliberately not recorded here -- it changes often and any number would be unverified.

### aider — Aider

- signup: https://aider.chat
- status: `documented`
- Open-source terminal coding agent. Free access exists but is not API-callable, so runner.py cannot drive it. Quota deliberately not recorded here -- it changes often and any number would be unverified.

### opencode — OpenCode

- signup: https://opencode.ai
- status: `documented`
- Open-source terminal coding agent. Free access exists but is not API-callable, so runner.py cannot drive it. Quota deliberately not recorded here -- it changes often and any number would be unverified.

### grok_web — xAI

- signup: https://grok.com
- status: `documented`
- Grok chat, free tier. Free access exists but is not API-callable, so runner.py cannot drive it. Quota deliberately not recorded here -- it changes often and any number would be unverified.

### deepseek_web — DeepSeek

- signup: https://chat.deepseek.com
- status: `documented`
- DeepSeek chat, free. Free access exists but is not API-callable, so runner.py cannot drive it. Quota deliberately not recorded here -- it changes often and any number would be unverified.

### qwen_chat — Alibaba

- signup: https://chat.qwen.ai
- status: `documented`
- Qwen chat, free. Free access exists but is not API-callable, so runner.py cannot drive it. Quota deliberately not recorded here -- it changes often and any number would be unverified.

### mistral_chat — Mistral

- signup: https://chat.mistral.ai
- status: `documented`
- Le Chat, free tier. Free access exists but is not API-callable, so runner.py cannot drive it. Quota deliberately not recorded here -- it changes often and any number would be unverified.

### perplexity — Perplexity

- signup: https://perplexity.ai
- status: `documented`
- Answer engine, free tier. Free access exists but is not API-callable, so runner.py cannot drive it. Quota deliberately not recorded here -- it changes often and any number would be unverified.

### kimi — Moonshot AI

- signup: https://kimi.com
- status: `documented`
- Kimi chat, free. Free access exists but is not API-callable, so runner.py cannot drive it. Quota deliberately not recorded here -- it changes often and any number would be unverified.

### zai_chat — Z.ai

- signup: https://chat.z.ai
- status: `documented`
- GLM chat, free tier. Free access exists but is not API-callable, so runner.py cannot drive it. Quota deliberately not recorded here -- it changes often and any number would be unverified.
