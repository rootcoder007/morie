# SPDX-License-Identifier: AGPL-3.0-or-later
"""Saved language-model settings: which route ``ask`` takes, and the address, key and model of each route.

The settings live in ``$XDG_CONFIG_HOME/morie/llm.json`` (default ``~/.config/morie/llm.json``),
a private (0600) file next to the shared ``credentials.json``. The same file and the same keys
are read by rmoriebricklayer (``rmbl config``), so a setting saved from one package applies to
all of them. The file is written only when the user saves a setting (``morie config set``,
``morie config setup`` or :func:`morie.llm.config`); nothing here writes it on its own.

Every setting has an environment variable too, and a variable that is set wins over the saved
value, so a one-off ``OLLAMA_MODEL=qwen3:8b morie ask ...`` still works. Older names morie
already read (``OLLAMA_BASE_URL``, ``MORIE_OLLAMA_MODEL``, ``LLM_API_BASE_URL``, ``LLM_API_KEY``,
``MORIE_API_MODEL``) are still accepted.

Language models are brought by the user: a provider's key (``<provider>.key``, saved privately,
never printed), a local Ollama, or an OpenAI-compatible server of one's own. Nothing is hosted
by the project. The MORIE key in ``credentials.json`` opens the curated tables only.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import stat
import warnings
from pathlib import Path

from .providers import IDS as PROVIDER_IDS
from .providers import PRESETS, RETIRED, model_env

ROUTES = ("auto", "own", "ollama", *PROVIDER_IDS)
_OFF = ("off", "none", "disabled")
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# key -> (environment variables, the first is the documented one; secret; help)
SPEC: dict[str, tuple[tuple[str, ...], bool, str]] = {
    "route": (
        ("MORIE_LLM_ROUTE",),
        False,
        "which route `ask` uses: auto (your own endpoint, then each provider with a key, then Ollama), "
        "own, ollama or a provider",
    ),
    "own.url": (
        ("MORIE_LLM_BASE_URL", "LLM_API_BASE_URL"),
        False,
        "your own OpenAI-compatible server, e.g. http://localhost:1234/v1 (LM Studio) or https://api.example.org/v1",
    ),
    "own.key": (("MORIE_LLM_API_KEY", "LLM_API_KEY"), True, "API key for your own server (sent as a Bearer token)"),
    "own.model": (("MORIE_LLM_MODEL", "MORIE_API_MODEL"), False, "model name on your own server"),
    "ollama.url": (
        ("OLLAMA_HOST", "OLLAMA_BASE_URL"),
        False,
        "your Ollama server, e.g. http://localhost:11434 or 192.168.1.20:11434; off to skip Ollama",
    ),
    "ollama.model": (
        ("OLLAMA_MODEL", "MORIE_OLLAMA_MODEL"),
        False,
        "Ollama model to use (default: the first one pulled)",
    ),
    "ollama.key": (("OLLAMA_API_KEY",), True, "API key for an Ollama server that wants one (ollama.com)"),
}
# one key and one model per provider (bring your own key), named as rmoriebricklayer names them
for _p in PRESETS:
    SPEC[f"{_p.id}.key"] = (
        tuple(e for e in (_p.key_env, _p.alt_env) if e),
        True,
        f"your {_p.label} API key (sent only to {_p.base})",
    )
    SPEC[f"{_p.id}.model"] = (
        (model_env(_p.id),),
        False,
        f"{_p.label} model to use (default {_p.default_model})"
        if _p.default_model
        else f"{_p.label} model to use (`morie models --route {_p.id}` lists them)",
    )
SPEC["r.mismatch"] = (
    ("MORIE_R_VERSION_MISMATCH",),
    False,
    "when the installed R package (rmorie or morie) is another version than this morie: "
    "warn (default: use it and say so once), quiet (use it silently) or strict (refuse)",
)
R_MISMATCH_MODES = ("warn", "quiet", "strict")
KEYS = tuple(SPEC)


def _normalise_key(key: str) -> str:
    """``own_url`` / ``OWN.URL`` / ``own-url`` -> ``own.url``; unknown keys raise ``KeyError``."""
    k = str(key).strip().lower().replace("_", ".").replace("-", ".")
    if k not in SPEC:
        raise KeyError(f"unknown setting {key!r} (one of: {', '.join(KEYS)})")
    return k


def config_path() -> Path:
    """``$XDG_CONFIG_HOME/morie/llm.json``, next to the shared credentials file."""
    from .hosted import credentials_path

    return credentials_path().parent / "llm.json"


def read_config() -> dict:
    """The saved settings; ``{}`` when the file is missing, unreadable or not a JSON object."""
    try:
        with open(config_path(), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, str)}


def _write_config(data: dict) -> Path:
    """Write the settings file owner-only (removed when nothing is left in it)."""
    path = config_path()
    if not data:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    # created private BEFORE anything is written, so no umask ever exposes a key
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
    os.replace(tmp, path)
    return path


def env_value(key: str) -> str | None:
    """The first of the setting's environment variables that is set (non-blank), else None."""
    for name in SPEC[_normalise_key(key)][0]:
        v = os.environ.get(name, "").strip()
        if v:
            return v
    return None


def saved_value(key: str) -> str | None:
    """The value saved in ``llm.json``."""
    k = _normalise_key(key)
    v = read_config().get(k, "").strip()
    return v or None


def value(key: str) -> str | None:
    """The setting in effect: its environment variable when set, else the saved value, else None."""
    return env_value(key) or saved_value(key)


def route() -> str:
    """The route ``ask`` is asked to take: ``auto`` unless MORIE_LLM_ROUTE or a saved route says otherwise."""
    r = (value("route") or "auto").lower()
    return r if r in ROUTES else "auto"


def is_off(v: str | None) -> bool:
    """True for the spellings that switch a route off (``off``, ``none``, ``disabled``)."""
    return bool(v) and v.strip().lower() in _OFF


def ollama_url() -> str | None:
    """The Ollama server: ollama.url (OLLAMA_HOST, OLLAMA_BASE_URL, saved), else localhost; None when off."""
    v = value("ollama.url") or DEFAULT_OLLAMA_URL
    if is_off(v):
        return None
    if not re.match(r"^https?://", v, re.I):
        v = "http://" + v  # OLLAMA_HOST is often host:port
    return v.rstrip("/")


def source(key: str) -> str:
    """Where a setting comes from right now: ``environment``, ``saved`` or ``default``."""
    k = _normalise_key(key)
    if env_value(k):
        return "environment"
    if saved_value(k):
        return "saved"
    if k in ("own.url", "own.key", "own.model"):
        from .hosted import read_credentials

        legacy = {"own.url": "api_base_url", "own.key": "api_key", "own.model": "api_model"}[k]
        if str(read_credentials().get(legacy, "")).strip():
            return "saved"  # attached with `morie provider set`
    return "default"


def mask(v: str | None) -> str:
    """How a key is displayed: only whether one is set. No character of it is ever printed."""
    return "set" if v else "(not set)"


def effective(key: str) -> str:
    """The value a setting has right now, the default spelled out; a key shows only as "set"."""
    k = _normalise_key(key)
    if k == "r.mismatch":
        return r_mismatch_mode()
    from . import llm

    if k == "route":
        return route()
    if k == "own.url":
        return llm._api_base_url() or "(not set)"
    if k == "own.key":
        v = llm._api_key()
        return mask(v)
    if k == "own.model":
        return llm._api_model()
    if k == "ollama.url":
        return ollama_url() or "off"
    if k == "ollama.model":
        return value("ollama.model") or "(the first one pulled)"
    if k == "ollama.key":
        v = value("ollama.key")
        return mask(v)
    from . import providers

    pid, _, what = k.partition(".")
    if what == "key":
        return mask(providers.key(pid))
    return providers.model(pid) or "(see `morie models --route " + pid + "`)"


def _check(key: str, v: str) -> str:
    """Validate (and tidy) one value before it is saved; raises ``ValueError``."""
    if key == "route":
        v = v.lower()
        if v == "hosted":
            raise ValueError(RETIRED)
        if v not in ROUTES:
            raise ValueError(f"route must be one of: {', '.join(ROUTES)}")
    elif key == "own.url":
        if not is_off(v) and not re.match(r"^https?://[^/\s]+", v, re.I):
            raise ValueError("own.url must be an http(s):// address, e.g. http://localhost:1234/v1")
        v = v.rstrip("/")
    elif key == "r.mismatch":
        v = v.lower()
        if v not in R_MISMATCH_MODES:
            raise ValueError(f"r.mismatch must be one of: {', '.join(R_MISMATCH_MODES)}")
    elif key == "ollama.url":
        if re.search(r"\s", v):
            raise ValueError("ollama.url must be an address such as http://localhost:11434")
        v = v.rstrip("/")
    return v


def table() -> list[dict[str, str]]:
    """One row per setting: key, value (a key only as "set"), source, env, help."""
    return [
        {
            "key": k,
            "value": effective(k),
            "source": source(k),
            "env": spec[0][0],
            "help": spec[2],
        }
        for k, spec in SPEC.items()
    ]


def format_table(rows: list[dict[str, str]] | None = None) -> str:
    """The settings as aligned text (what ``morie config`` prints)."""
    rows = table() if rows is None else rows
    out = []
    for r in rows:
        tag = "" if r["source"] == "default" else f"({r['source']})"
        out.append(f"  {r['key']:<17} {r['value']:<44} {tag}".rstrip())
    return "\n".join(out)


class ConfigTable(list):
    """The list of setting rows :func:`config` returns; prints as an aligned table."""

    def __repr__(self) -> str:
        return format_table(list(self))

    __str__ = __repr__


def config(settings: dict | None = None, /, **kwargs) -> ConfigTable:
    """Show or save the language-model settings.

    With no arguments nothing is written and the table of settings is returned. Pass settings as
    keyword arguments with ``_`` for the dot (``own_url=...``) or as a dict with the dotted keys
    (``{"own.url": ...}``). ``None`` or ``""`` removes a saved setting.

    Keys: route (auto, own, ollama or a provider id), own.url, own.key, own.model, ollama.url,
    ollama.model, ollama.key, and ``<provider>.key`` / ``<provider>.model`` for each provider
    (openai, anthropic, gemini, openrouter, groq, ...; ``morie.providers.IDS``).

    Returns
    -------
    ConfigTable
        A list of dicts (key, value, source, env, help), one per setting.

    Examples
    --------
    >>> from morie.llm import config
    >>> config(route="anthropic", anthropic_model="claude-opus-5-5")  # doctest: +SKIP
    >>> config(ollama_url="http://192.168.1.20:11434", ollama_model="qwen3:8b")  # doctest: +SKIP
    >>> config(route=None)  # back to the automatic order  # doctest: +SKIP
    """
    items: dict[str, object] = {}
    for k, v in {**(settings or {}), **kwargs}.items():
        items[_normalise_key(k)] = v
    if items:
        data = read_config()
        changed: list[str] = []
        for k, v in items.items():
            text = "" if v is None else str(v).strip()
            if text:
                data[k] = _check(k, text)
            else:
                data.pop(k, None)
            changed.append(k)
        _write_config(data)
        _reset_caches()
        shadowed = [SPEC[k][0][0] for k in changed if env_value(k)]
        if shadowed:
            warnings.warn(
                f"{', '.join(shadowed)} is set in the environment and still wins over the saved value",
                stacklevel=2,
            )
    return ConfigTable(table())


def r_mismatch_mode() -> str:
    """warn, quiet or strict: what the R-backed commands do when the R package is another version."""
    v = (value("r.mismatch") or "").strip().lower()
    return v if v in R_MISMATCH_MODES else "warn"


def config_get(key: str) -> str:
    """The value one setting has right now (a key only as "set"; no character of it is returned)."""
    return effective(key)


def config_unset(*keys: str) -> ConfigTable:
    """Remove saved settings (the environment and defaults apply again)."""
    return config({k: None for k in keys})


def _reset_caches() -> None:
    """Forget cached probes so a changed setting applies in this process at once."""
    with contextlib.suppress(Exception):
        from . import llm

        llm._reset_route_cache()


# ---------------------------------------------------------------------------
# `morie config ...`
# ---------------------------------------------------------------------------

CLI_ACTIONS = ("show", "list", "help", "get", "set", "unset", "setup", "path")
_USAGE = "usage: morie config [show | help | get KEY | set KEY VALUE | unset KEY | setup | path]"


def _examples() -> str:
    return "\n".join(
        [
            "Examples:",
            "  morie config set anthropic.key                         prompts for the key (not echoed)",
            "  morie config set route anthropic                       always use it",
            "  morie config set ollama.url http://192.168.1.20:11434  Ollama on another machine",
            "  morie config set ollama.model qwen3:8b",
            "  morie config set own.url http://localhost:1234/v1      LM Studio, vLLM, llama.cpp ...",
            "  morie config set own.key                               prompts for the key (not echoed)",
            "  morie config unset route                               back to the automatic order",
            '  morie ask --route groq "..."                           one question, one route',
        ]
    )


def help_text() -> str:
    """What every setting means (``morie config help``)."""
    lines = []
    for k, (envs, _secret, text) in SPEC.items():
        lines.append(f"  {k:<17} {text}")
        lines.append(f"  {'':<17} (environment variable {' or '.join(envs)})")
    lines += [
        "",
        f"Saved in {config_path()} (private, 0600). A variable that is set wins over a saved value.",
        "",
        _examples(),
    ]
    return "\n".join(lines)


def cli(action: str | None, rest: list[str] | None = None, *, out=print, ask=input, secret_ask=None) -> int:
    """``morie config ACTION ...``; returns the exit code. ``out``/``ask`` are seams for tests."""
    import getpass

    rest = list(rest or [])
    action = (action or "show").lower()
    secret_ask = secret_ask or getpass.getpass

    def need_key() -> str | None:
        if not rest:
            out(f"usage: morie config {action} KEY{' VALUE' if action == 'set' else ''}  (keys: {', '.join(KEYS)})")
            return None
        try:
            return _normalise_key(rest[0])
        except KeyError as exc:
            out(f"morie config: {exc.args[0]}")
            return None

    if action in ("show", "list"):
        out(format_table())
        out(
            f"\nSaved in {config_path()}. An environment variable that is set wins over a saved value.\n"
            "Change one:  morie config set route anthropic\n"
            "Walk through all of them:  morie config setup\n"
            "What each one means:  morie config help"
        )
        return 0
    if action == "help":
        out(help_text())
        return 0
    if action == "path":
        out(str(config_path()))
        return 0
    if action == "get":
        k = need_key()
        if k is None:
            return 2
        out(effective(k))
        return 0
    if action in ("set", "unset"):
        k = need_key()
        if k is None:
            return 2
        if action == "set":
            v = " ".join(rest[1:]).strip()
            if not v and SPEC[k][1]:
                v = secret_ask(f"{k}: ").strip()
            if not v:
                out(f"usage: morie config set {k} VALUE")
                return 2
        else:
            v = None
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                config({k: v})
            for w in caught:
                out(f"morie config: {w.message}")
        except (ValueError, RuntimeError, OSError) as exc:
            out(f"morie config: {exc}")
            return 1
        out(f"{k} = {effective(k)}")
        return 0
    if action == "setup":
        return setup(out=out, ask=ask, secret_ask=secret_ask)
    out(_USAGE)
    return 2


def setup(*, out=print, ask=input, secret_ask=None) -> int:
    """``morie config setup``: one question per route; Enter keeps what is there."""
    import getpass

    secret_ask = secret_ask or getpass.getpass

    def q(text: str, current: str = "") -> str:
        a = ask(f"{text}{f' [{current}]' if current else ''}: ").strip()
        return a or current

    def save(**kw) -> None:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                config(kw)
            for w in caught:
                out(f"  note: {w.message}")
        except (ValueError, RuntimeError, OSError) as exc:
            out(f"  not saved: {exc}")

    out("Language-model setup: bring your own key, or run a model locally. Enter keeps the value in [brackets].\n")
    out(f"Providers: {', '.join(PROVIDER_IDS)}")
    r = q("Which should `ask` use: auto, own, ollama or a provider", route()).lower()
    if r == "hosted":
        out(f"  {RETIRED}; using auto")
        r = "auto"
    if r not in ROUTES:
        out(f"  '{r}' is not a route; using auto")
        r = "auto"
    save(route=None if r == "auto" else r)
    from . import providers

    pids = [r] if r in PROVIDER_IDS else []
    if r == "auto":
        pid = q("\nA provider whose key you have (Enter to skip)", "").lower()
        if pid and pid not in PROVIDER_IDS:
            out(f"  '{pid}' is not a provider; skipped")
        elif pid:
            pids = [pid]
    for pid in pids:
        label = providers.preset(pid).label
        k = secret_ask(f"{label} key (Enter to skip or to keep): ").strip()
        if k:
            save(**{f"{pid}.key": k})
        if providers.key(pid):
            listed = providers.list_models(pid)
            if listed:
                out(f"Models: {', '.join(listed[:30])}{' ...' if len(listed) > 30 else ''}")
            m = q(f"{label} model", providers.model(pid))
            if m and m != providers.preset(pid).default_model:
                save(**{f"{pid}.model": m})
    if r in ("auto", "ollama"):
        u = q("\nOllama address (off to skip Ollama)", effective("ollama.url"))
        save(**{"ollama.url": None if u == DEFAULT_OLLAMA_URL else u})
        m = q("Ollama model (Enter for the first pulled)", value("ollama.model") or "")
        save(**{"ollama.model": m or None})
    if r in ("auto", "own"):
        u = q("\nYour own OpenAI-compatible server (Enter to skip)", value("own.url") or "")
        save(**{"own.url": u or None})
        if u:
            save(**{"own.model": q("Model on that server", value("own.model") or "") or None})
            k = secret_ask("API key for it (Enter for none or to keep): ").strip()
            if k:
                save(**{"own.key": k})
    out(f"\nSaved in {config_path()}.\n")
    from . import llm

    for row in llm.route_status():
        out(f"  {row['route']:<20} {row['status']:<14} {row['detail']}")
    out(f"\n  {llm.route_summary()}")
    return 0
