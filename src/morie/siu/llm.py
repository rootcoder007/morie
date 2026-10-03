# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bring-your-own language model -- the Python twin of the C++ ``siu::llm`` layer.

Nothing is hardcoded: no host, no model, no key. A :class:`Backend` points at
ANY server speaking one of two protocols, or at your own function:

* ``api="ollama"`` -- ``POST {base}/api/chat``, ``GET {base}/api/tags``
  (Ollama, local or remote, or a tunnelled gateway);
* ``api="openai"`` -- ``POST {base}/chat/completions``, ``GET {base}/models``
  with ``base`` ending in ``/v1`` (llama.cpp ``llama-server``, vLLM, LM Studio,
  LocalAI, text-generation-webui, TGI, Jan, OpenRouter, OpenAI, ...);
* ``chat=`` -- your own ``callable(model, prompt) -> str`` (a home-made model).

Empty fields fall back to the environment exactly as the C++ core does:
``MORIE_LLM_API`` (default ``ollama``); ``MORIE_LLM_BASE``, else
``OLLAMA_HOST`` / ``OLLAMA_BASE_URL`` (ollama) or ``OPENAI_BASE_URL`` /
``LLM_API_BASE_URL`` (openai), else ``http://localhost:11434`` or
``http://localhost:8080/v1``; an openai base gets ``/v1`` appended unless it
already ends in ``/v1``; ``MORIE_LLM_KEY``, else ``OLLAMA_API_KEY`` or
``OPENAI_API_KEY`` / ``LLM_API_KEY``. HTTP 429/503 are retried with linear
backoff (2 s, 4 s, ... six times), as in the C++ ``http`` layer.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, replace

__all__ = ["Backend", "chat", "default_model", "list_models", "resolve"]

ChatFn = Callable[[str, str], str]


@dataclass(frozen=True)
class Backend:
    """Where the models live. ``chat`` (your own function) takes precedence over HTTP."""

    api: str = ""
    base: str = ""
    key: str = ""
    timeout: float = 300.0
    temperature: float = 0.0
    chat: ChatFn | None = None


def _env(*names: str) -> str:
    for n in names:
        v = os.environ.get(n, "")
        if v:
            return v
    return ""


def resolve(b: Backend | None = None, **kw) -> Backend:
    """Fill empty ``api``/``base``/``key`` from the environment and normalise the base URL.

    Examples
    --------
    >>> resolve(api="openai", base="https://openrouter.ai/api").base
    'https://openrouter.ai/api/v1'
    >>> resolve(api="ollama", base="gpu-box:11434/", key="k").base
    'http://gpu-box:11434'
    """
    b = replace(b or Backend(), **kw)
    api = b.api or _env("MORIE_LLM_API") or "ollama"
    if api not in ("ollama", "openai"):
        raise ValueError('api must be "ollama" or "openai"')
    base = b.base or _env("MORIE_LLM_BASE")
    if not base:
        base = (
            _env("OLLAMA_HOST", "OLLAMA_BASE_URL") if api == "ollama" else _env("OPENAI_BASE_URL", "LLM_API_BASE_URL")
        )
    if not base:
        base = "http://localhost:11434" if api == "ollama" else "http://localhost:8080/v1"
    if not base.startswith("http"):
        base = "http://" + base
    base = base.rstrip("/")
    if api == "openai" and not base.endswith("/v1"):
        base += "/v1"
    key = b.key or (
        _env("MORIE_LLM_KEY", "OLLAMA_API_KEY")
        if api == "ollama"
        else _env("MORIE_LLM_KEY", "OPENAI_API_KEY", "LLM_API_KEY")
    )
    return replace(b, api=api, base=base, key=key)


def _request(url: str, *, body: bytes | None, key: str, timeout: float) -> bytes:
    headers = {"User-Agent": "morie-siu/1.0"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if key:
        headers["Authorization"] = "Bearer " + key
    for attempt in range(7):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST" if body is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < 6:
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"http status {e.code} for {url}") from e
    raise RuntimeError(f"http retries exhausted for {url}")


def list_models(b: Backend | None = None, **kw) -> list[str]:
    """Model names the server offers (``[]`` on failure or for a custom ``chat``)."""
    b = replace(b or Backend(), **kw)
    if b.chat is not None:
        return []
    b = resolve(b)
    try:
        if b.api == "ollama":
            body = json.loads(_request(b.base + "/api/tags", body=None, key=b.key, timeout=10))
            return [m.get("name") or m.get("model") for m in body.get("models", []) if m.get("name") or m.get("model")]
        body = json.loads(_request(b.base + "/models", body=None, key=b.key, timeout=10))
        return [m["id"] for m in body.get("data", []) if "id" in m]
    except Exception:  # noqa: BLE001 -- a dead server lists no models
        return []


def chat(b: Backend | None, model: str, prompt: str) -> str:
    """One user-turn chat; returns the assistant text (raises on transport errors)."""
    b = b or Backend()
    if b.chat is not None:
        return str(b.chat(model, prompt))
    b = resolve(b)
    messages = [{"role": "user", "content": prompt}]
    if b.api == "ollama":
        req = {"model": model, "stream": False, "options": {"temperature": b.temperature}, "messages": messages}
        body = json.loads(_request(b.base + "/api/chat", body=json.dumps(req).encode(), key=b.key, timeout=b.timeout))
        return (body.get("message") or {}).get("content", "")
    req = {"model": model, "stream": False, "temperature": b.temperature, "messages": messages}
    body = json.loads(
        _request(b.base + "/chat/completions", body=json.dumps(req).encode(), key=b.key, timeout=b.timeout)
    )
    choices = body.get("choices") or []
    if not choices:
        return ""
    content = (choices[0].get("message") or {}).get("content")
    return content if isinstance(content, str) else ""


def default_model(b: Backend | None = None, **kw) -> str:
    """``$MORIE_LLM_MODEL``, else ``$OLLAMA_MODEL``, else the first model the backend lists (``""`` if none)."""
    m = _env("MORIE_LLM_MODEL", "OLLAMA_MODEL")
    if m:
        return m
    ms = list_models(b, **kw)
    return ms[0] if ms else ""
