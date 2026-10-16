# SPDX-License-Identifier: AGPL-3.0-or-later
"""Language-model providers you bring your own key for: the same table rmoriebricklayer uses.

Each provider is read from its usual key variable (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ...)
or from a key saved with ``morie config set <provider>.key`` (``$XDG_CONFIG_HOME/morie/llm.json``,
owner-only, the file rmoriebricklayer's ``rmbl config`` writes too). The model is
``MORIE_<PROVIDER>_MODEL`` or ``<provider>.model``, else the provider's default. Anthropic's
Messages API and Gemini's ``generateContent`` are spoken natively; every other provider is
OpenAI-compatible. Nothing is hosted by the project: a prompt goes only where a key is set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class Preset:
    id: str
    label: str
    base: str
    api: str  # "openai", "anthropic" or "gemini"
    key_env: str
    alt_env: str = ""
    default_model: str = ""
    models: bool = True  # lists its models at a models endpoint


PRESETS: tuple[Preset, ...] = (
    Preset("openai", "OpenAI", "https://api.openai.com/v1", "openai", "OPENAI_API_KEY"),
    Preset(
        "anthropic",
        "Anthropic",
        "https://api.anthropic.com",
        "anthropic",
        "ANTHROPIC_API_KEY",
        default_model="claude-opus-5-5",
    ),
    Preset(
        "gemini",
        "Google Gemini",
        "https://generativelanguage.googleapis.com",
        "gemini",
        "GEMINI_API_KEY",
        alt_env="GOOGLE_API_KEY",
    ),
    Preset("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "openai", "OPENROUTER_API_KEY"),
    Preset("groq", "Groq", "https://api.groq.com/openai/v1", "openai", "GROQ_API_KEY"),
    Preset("together", "Together AI", "https://api.together.xyz/v1", "openai", "TOGETHER_API_KEY"),
    Preset("fireworks", "Fireworks AI", "https://api.fireworks.ai/inference/v1", "openai", "FIREWORKS_API_KEY"),
    Preset("deepseek", "DeepSeek", "https://api.deepseek.com/v1", "openai", "DEEPSEEK_API_KEY"),
    Preset("mistral", "Mistral", "https://api.mistral.ai/v1", "openai", "MISTRAL_API_KEY"),
    Preset("xai", "xAI", "https://api.x.ai/v1", "openai", "XAI_API_KEY"),
    Preset("cerebras", "Cerebras", "https://api.cerebras.ai/v1", "openai", "CEREBRAS_API_KEY"),
    Preset(
        "perplexity",
        "Perplexity",
        "https://api.perplexity.ai",
        "openai",
        "PERPLEXITY_API_KEY",
        default_model="sonar-pro",
        models=False,
    ),
    Preset(
        "huggingface",
        "Hugging Face",
        "https://router.huggingface.co/v1",
        "openai",
        "HF_TOKEN",
        alt_env="HUGGINGFACE_HUB_TOKEN",
    ),
    Preset("opencode", "OpenCode Zen", "https://opencode.ai/zen/v1", "openai", "OPENCODE_API_KEY"),
    Preset(
        "cohere", "Cohere", "https://api.cohere.ai/compatibility/v1", "openai", "COHERE_API_KEY", alt_env="CO_API_KEY"
    ),
    Preset("nvidia", "NVIDIA NIM", "https://integrate.api.nvidia.com/v1", "openai", "NVIDIA_API_KEY"),
)
IDS: tuple[str, ...] = tuple(p.id for p in PRESETS)
_BY_ID = {p.id: p for p in PRESETS}

RETIRED = (
    "the hosted MORIE language-model tier is retired: bring your own key (a provider's key such as "
    "ANTHROPIC_API_KEY or OPENAI_API_KEY), a local Ollama, or your own OpenAI-compatible server "
    "(`morie config setup` walks through it)"
)


def preset(pid: str) -> Preset:
    """The preset for a provider id (``KeyError`` for an unknown one)."""
    return _BY_ID[pid]


def model_env(pid: str) -> str:
    """The variable that names a provider's model, e.g. ``MORIE_ANTHROPIC_MODEL``."""
    return f"MORIE_{pid.upper()}_MODEL"


def _saved(key: str) -> str | None:
    from .llm_config import saved_value

    return saved_value(key)


def key(pid: str) -> str | None:
    """A provider's key: its variable, the alternative spelling, else the saved ``<id>.key``."""
    p = preset(pid)
    for env in (p.key_env, p.alt_env):
        v = os.environ.get(env, "").strip() if env else ""
        if v:
            return v
    return _saved(f"{pid}.key")


def model(pid: str) -> str:
    """A provider's model: ``MORIE_<ID>_MODEL``, else the saved ``<id>.model``, else its default."""
    return os.environ.get(model_env(pid), "").strip() or _saved(f"{pid}.model") or preset(pid).default_model


def with_key() -> list[str]:
    """The providers whose key is set, in the table's order."""
    return [p.id for p in PRESETS if key(p.id)]


def headers(api: str, api_key: str | None) -> dict[str, str]:
    """The request headers a provider needs: its key in its own header."""
    if api == "anthropic":
        h = {"anthropic-version": "2023-06-01"}
        if api_key:
            h["x-api-key"] = api_key
        return h
    if api == "gemini":
        return {"x-goog-api-key": api_key} if api_key else {}
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    sys_txt = "\n\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
    return sys_txt, [m for m in messages if m.get("role") != "system"]


def anthropic_body(model_name: str, messages: list[dict], max_tokens: int = 4096) -> dict:
    """The Messages API body: the system text apart, the turns in order (long instructions marked
    for Anthropic's prompt cache, as rmoriebricklayer does)."""
    sys_txt, rest = _split_system(messages)
    body: dict = {
        "model": model_name,
        "max_tokens": int(max_tokens),
        "messages": [
            {"role": "assistant" if m.get("role") == "assistant" else "user", "content": str(m.get("content", ""))}
            for m in rest
        ],
    }
    if sys_txt:
        body["system"] = (
            sys_txt
            if len(sys_txt) < 4000
            else [{"type": "text", "text": sys_txt, "cache_control": {"type": "ephemeral"}}]
        )
    return body


def anthropic_text(js: dict) -> str:
    return "\n".join(str(b.get("text", "")) for b in js.get("content") or [] if b.get("type") == "text")


def gemini_body(messages: list[dict], max_tokens: int = 4096) -> dict:
    """The ``generateContent`` body: the system text as ``systemInstruction``, turns as ``contents``."""
    sys_txt, rest = _split_system(messages)
    body: dict = {
        "contents": [
            {
                "role": "model" if m.get("role") == "assistant" else "user",
                "parts": [{"text": str(m.get("content", ""))}],
            }
            for m in rest
        ],
        "generationConfig": {"maxOutputTokens": int(max_tokens)},
    }
    if sys_txt:
        body["systemInstruction"] = {"parts": [{"text": sys_txt}]}
    return body


def gemini_text(js: dict) -> str:
    cands = js.get("candidates") or [{}]
    parts = (cands[0].get("content") or {}).get("parts") or []
    return "\n".join(str(p.get("text", "")) for p in parts if "text" in p)


def chat_native(
    api: str, base: str, api_key: str | None, model_name: str, messages: list[dict], timeout: float = 120.0
) -> str:
    """One answer from Anthropic or Gemini, in their own wire format. Raises ``httpx.HTTPError``
    (a refused key, an unknown model, a dead connection) for the caller's fallback to handle."""
    if api == "anthropic":
        url, body, parse = f"{base}/v1/messages", anthropic_body(model_name, messages), anthropic_text
    elif api == "gemini":
        url, body, parse = f"{base}/v1beta/models/{model_name}:generateContent", gemini_body(messages), gemini_text
    else:
        raise ValueError(f"not a native API: {api}")
    resp = httpx.post(url, json=body, headers=headers(api, api_key), timeout=timeout)
    resp.raise_for_status()
    return parse(resp.json())


def list_models(pid: str, timeout: float = 10.0) -> list[str]:
    """The models a provider offers this key (empty when it publishes no list or does not answer)."""
    p, k = preset(pid), key(pid)
    if not k or not p.models:
        return [m for m in (model(pid),) if m]
    try:
        if p.api == "gemini":
            r = httpx.get(f"{p.base}/v1beta/models", headers=headers("gemini", k), timeout=timeout)
            r.raise_for_status()
            return [str(m.get("name", "")).removeprefix("models/") for m in r.json().get("models", [])]
        url = f"{p.base}/v1/models" if p.api == "anthropic" else f"{p.base}/models"
        r = httpx.get(url, headers=headers(p.api, k), timeout=timeout)
        r.raise_for_status()
        return [str(m.get("id", "")) for m in r.json().get("data", []) if m.get("id")]
    except (httpx.HTTPError, ValueError):
        return []
