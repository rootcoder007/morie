"""The shared settings file, the data service address, and your own model endpoint.

The curated tables at data.rmorie.com need no sign-in and no key, and none is read or sent
(a key an earlier version stored stays in the file, unread). No language model is hosted by
the project either: a model is brought by the user (:mod:`morie.providers`). What remains
here is ``$XDG_CONFIG_HOME/morie/credentials.json``, kept 0600, which holds the endpoint
``morie provider set`` saves, and the data service's address from the signed services
document (``morie.services``).
"""

from __future__ import annotations

import contextlib
import json
import os
import stat
from pathlib import Path

DEFAULT_DATA_BASE_URL = "https://data.rmorie.com"


def hosted_base_url() -> str | None:
    """The data service (the services document's ``data.base_url``), or None when the
    document switches it off."""
    from . import services

    d = services.data()
    if d.get("mode") != "key" or not d.get("base_url"):
        return None
    return str(d["base_url"]).rstrip("/")


def credentials_path() -> Path:
    """``$XDG_CONFIG_HOME/morie/credentials.json`` (shared with the R package)."""
    base = os.environ.get("XDG_CONFIG_HOME", "").strip() or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "morie" / "credentials.json"


def read_credentials() -> dict:
    """The stored credentials as a dict; ``{}`` when the file is missing, unreadable or not a JSON object."""
    try:
        with open(credentials_path(), encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_credentials(data: dict) -> Path:
    """Write the credentials file with owner-only permissions."""
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
    os.replace(tmp, path)
    return path


def _say(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Your own endpoint: `morie provider set|show|unset`
# ---------------------------------------------------------------------------

_PROVIDER_KEYS = ("api_base_url", "api_key", "api_model")


def provider_set(base_url: str, key: str, model: str | None = None, echo=_say) -> dict:
    """Attach an OpenAI-compatible endpoint (chat completions at BASE_URL/chat/completions).

    Works for OpenAI, Anthropic's compatibility endpoint (https://api.anthropic.com/v1),
    OpenRouter, Mistral, Groq, a local LM Studio / vLLM / llama.cpp server, or any
    other server that speaks that API. Environment variables LLM_API_BASE_URL,
    LLM_API_KEY and MORIE_API_MODEL still take precedence when set.
    """
    base_url = base_url.strip().rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("base URL must start with http:// or https://")
    if not key.strip():
        raise ValueError("the key is empty")
    data = read_credentials()
    data["api_base_url"] = base_url
    data["api_key"] = key.strip()
    if model and model.strip():
        data["api_model"] = model.strip()
    else:
        data.pop("api_model", None)
    path = write_credentials(data)
    echo(
        f"Endpoint attached: {base_url}"
        + (f" (model {data.get('api_model')})" if data.get("api_model") else "")
        + f"; stored in {path}"
    )
    return {k: data.get(k) for k in _PROVIDER_KEYS}


def provider_show(echo=_say) -> dict:
    """Describe the attached OpenAI-compatible endpoint without printing its key.

    Args:
        echo: where the description goes (default: stdout).

    Returns:
        The stored ``api_base_url``, ``api_model`` and ``api_key`` entries (``None`` where unset).
    """
    data = read_credentials()
    base, model = data.get("api_base_url"), data.get("api_model")
    if not base:
        echo("No endpoint attached. Attach one with: morie provider set --base-url URL --key KEY [--model NAME]")
    else:
        key = str(data.get("api_key", ""))
        echo(
            f"Endpoint: {base}\nModel:    {model or 'server default'}\nKey:      stored ({len(key)} characters; never printed)"
        )
    return {k: data.get(k) for k in _PROVIDER_KEYS}


def provider_unset(echo=_say) -> bool:
    """Detach the OpenAI-compatible endpoint from the stored credentials.

    Other stored credentials are kept; the file is removed only when
    nothing else is left in it, and nothing is written when no endpoint
    was attached.

    Args:
        echo: where the outcome line goes (default: stdout).

    Returns:
        True if an endpoint was attached.
    """
    data = read_credentials()
    had = any(k in data for k in _PROVIDER_KEYS)
    for k in _PROVIDER_KEYS:
        data.pop(k, None)
    if data:
        write_credentials(data)
    else:
        with contextlib.suppress(FileNotFoundError):
            credentials_path().unlink()
    echo("Endpoint detached." if had else "No endpoint was attached.")
    return had


def provider_line() -> str | None:
    """One line for `morie models` when an endpoint is attached or set in the environment."""
    from .llm import _api_base_url, _api_key, _api_model

    base = _api_base_url()
    if not base or not _api_key():
        return None
    return f"Your endpoint ({base}): model {_api_model()}"
