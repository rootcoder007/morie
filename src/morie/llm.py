"""Ollama-first LLM integration layer for the MORIE package.

Provides a provider chain that attempts local Ollama inference first, then
Gemini (Google), then a generic OpenAI-compatible endpoint (e.g. Qwen via
OpenRouter, GPT-OSS models via Together/Groq), then the official OpenAI API, and finally a
local help-text fallback that requires no network access.

HTTP-based providers use ``httpx`` against OpenAI-compatible endpoints.

Environment Variables
---------------------
OLLAMA_BASE_URL : str
    Base URL for a running Ollama instance.  Default: ``http://localhost:11434``
GEMINI_API_KEY : str
    Google AI Studio API key.  Free-tier keys work for development.
    Model defaults to ``gemini-2.0-flash``.
GEMINI_MODEL : str
    Override the Gemini model (e.g. ``gemini-1.5-pro``).  Optional.
LLM_API_BASE_URL : str
    Base URL for any OpenAI-compatible API (e.g., OpenRouter, Together, Groq).
    Use this to point at Qwen, Mistral, GPT-OSS, or any hosted model.
LLM_API_KEY : str
    API key for the endpoint at ``LLM_API_BASE_URL``.
OPENAI_API_KEY : str
    API key for the official OpenAI API at ``https://api.openai.com``.
MORIE_LLM_ROUTE : str
    ``auto`` (default), ``own``, ``ollama`` or ``hosted``: force the route ``ask`` takes.
OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_API_KEY : str
    The Ollama server (``OLLAMA_BASE_URL`` is still read), its model and an optional key.
MORIE_LLM_BASE_URL, MORIE_LLM_API_KEY, MORIE_LLM_MODEL : str
    Your own OpenAI-compatible server (the ``LLM_API_*`` names above are still read).

Every one of these can be saved instead with ``morie config set KEY VALUE`` or
:func:`config` (``$XDG_CONFIG_HOME/morie/llm.json``); a variable that is set wins over
the saved value. ``morie config help`` lists them, ``morie doctor`` says which route
``ask`` takes.

Provider priority (auto-detected at runtime, ``route = auto``):
    1. Ollama    -- local, private, no API key needed; only when it has a model
                    (one pulled, or ``OLLAMA_MODEL`` set): a server with nothing
                    pulled is skipped
    2. Gemini    -- Google AI, generous free tier
    3. API       -- your own OpenAI-compatible endpoint (Qwen, GPT-OSS, Groq, LM Studio, ...)
    4. OpenAI    -- official OpenAI API
    5. hosted    -- the hosted MORIE tier, after ``morie login``
    6. local     -- static help text, no network required

References
----------
* Ollama API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
* Gemini OpenAI-compatible API: https://ai.google.dev/gemini-api/docs/openai
* OpenAI Chat Completions API: https://platform.openai.com/docs/api-reference/chat
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

from . import llm_config as _cfg
from .cpads import cpads_contract
from .llm_config import config, config_get, config_path, config_unset  # noqa: F401 -- public API
from .modules import MODULE_SPECS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = ""  # Auto-detected from running Ollama instance
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_API_MODEL = "google/gemma-3-27b-it"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

OPENAI_BASE_URL = "https://api.openai.com"
# Gemini exposes an OpenAI-compatible endpoint; no extra SDK required.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

_PROVIDER_OLLAMA = "ollama"
_PROVIDER_HOSTED = "hosted"
_PROVIDER_GEMINI = "gemini"
_PROVIDER_API = "api"
_PROVIDER_OPENAI = "openai"
_PROVIDER_LOCAL = "local"

# Timeout for the quick health-check probe (seconds).
_PROBE_TIMEOUT = 2.0

# Timeout for actual generation requests (seconds).
_REQUEST_TIMEOUT = 120.0

# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

_MORIE_SYSTEM_PROMPT_TEMPLATE = """\
You are the MORIE agent for methods for observational inference and robust analysis of interventions in sociolegal studies.

MORIE is a Python+R terminal IDE for Canadian public health data analysis, \
causal inference, and reproducible research. Install: pip install morie

TUI keys: c=Chat p=Pipeline d=Doctor i=Datasets h=Help s=Stats e=REPL q=Quit
Chat commands: /run /list /doctor /profile /inspect /verify /agent /help /clear
REPL: ?question=AI !cmd=shell R>code=R. Helpers: load() head() describe() cols()
CLI: morie list-modules, morie run-module <name>, morie pipeline --all -y
CLI: morie list-datasets, morie doctor, morie selftest, morie ask "question"

32 built-in datasets: CPADS, CCS, CSADS, CSUS, HealthInfobase, CIHI.
Load: load('cpads') in REPL or from morie.data import load_dataset

48 stats commands: ttest anova chi2 corr regression pscore ipw aipw ate \
cohend kaplanmeier coxph did rddesign ivreg vif and more (press s).

21 modules: data-wrangling descriptive-statistics frequentist-inference \
bayesian-inference propensity-scores causal-estimators treatment-effects \
ebac-core figures tables final-report and more.

Debug: press d for Doctor, b for logs, morie selftest for smoke tests.

Give practical answers with specific MORIE commands. Be explicit about \
assumptions and limitations.

{context_block}
"""

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------


def _ollama_base_url() -> str | None:
    """The Ollama server: OLLAMA_HOST, OLLAMA_BASE_URL or the saved ollama.url, else localhost; None when off."""
    return _cfg.ollama_url()


def _ollama_key() -> str | None:
    """OLLAMA_API_KEY, else the saved ollama.key (for an Ollama server behind a gateway)."""
    return _cfg.value("ollama.key")


_ollama_model_cached: str | None = None
_UNPROBED: list = []  # sentinel: the tags cache has not been filled yet
_ollama_tags_cached: list[dict] | None = _UNPROBED


def _ollama_tags(timeout: float = _PROBE_TIMEOUT) -> list[dict] | None:
    """The models the Ollama server has (``GET /api/tags``): a list of ``{"name", "size"}``,
    ``[]`` for a server with nothing pulled, ``None`` when nothing answers. Cached per process."""
    global _ollama_tags_cached
    if _ollama_tags_cached is not _UNPROBED:
        return _ollama_tags_cached
    base = _ollama_base_url()
    tags: list[dict] | None = None
    if base:
        key = _ollama_key()
        try:
            resp = httpx.get(
                f"{base}/api/tags",
                headers={"Authorization": f"Bearer {key}"} if key else None,
                timeout=timeout,
            )
            if resp.status_code < 400:
                models = (resp.json() or {}).get("models") or []
                tags = [
                    {"name": str(m.get("name") or m.get("model")), "size": m.get("size") or 0}
                    for m in models
                    if isinstance(m, dict) and (m.get("name") or m.get("model"))
                ]
        except Exception:  # noqa: BLE001 - not running, refused, not JSON
            tags = None
    _ollama_tags_cached = tags
    return tags


def _ollama_model() -> str:
    """Return the Ollama model to use.

    Priority:
    1. OLLAMA_MODEL / MORIE_OLLAMA_MODEL, else the saved ollama.model (``morie config``)
    2. The server's largest ``perseus*`` model, else the first model it has
    3. Empty string (no model available)
    """
    global _ollama_model_cached
    chosen = _cfg.value("ollama.model")
    if chosen:
        return chosen
    if _ollama_model_cached is not None:
        return _ollama_model_cached
    tags = _ollama_tags() or []
    perseus = sorted((t for t in tags if t["name"].startswith("perseus")), key=lambda t: t["size"], reverse=True)
    pick = perseus[0]["name"] if perseus else (tags[0]["name"] if tags else DEFAULT_OLLAMA_MODEL)
    _ollama_model_cached = pick
    return pick


def _reset_route_cache() -> None:
    """Forget the cached Ollama probe and model so changed settings apply at once."""
    global _ollama_cached, _ollama_tags_cached, _ollama_model_cached
    _ollama_cached = None
    _ollama_tags_cached = _UNPROBED
    _ollama_model_cached = None


def _stored_provider() -> dict:
    """The endpoint attached with `morie provider set` (shared credentials file)."""
    try:
        from .hosted import read_credentials

        data = read_credentials()
    except Exception:
        return {}
    return {k: v for k, v in data.items() if k in ("api_base_url", "api_key", "api_model") and v}


def _api_base_url() -> str | None:
    """The generic OpenAI-compatible base URL: LLM_API_BASE_URL, else MORIE_LLM_BASE_URL (the
    name rmoriebricklayer, rmoriedata and rmorie read), else the saved own.url (``morie config``),
    else the endpoint attached with ``morie provider set``. ``off`` switches it off."""
    url = (
        os.environ.get("LLM_API_BASE_URL", "").strip()
        or os.environ.get("MORIE_LLM_BASE_URL", "").strip()
        or _cfg.saved_value("own.url")
        or str(_stored_provider().get("api_base_url", "")).strip()
    )
    if not url or _cfg.is_off(url):
        return None
    return url.rstrip("/")


def _api_key() -> str | None:
    """The key for that endpoint: LLM_API_KEY, else MORIE_LLM_API_KEY, else the saved own.key,
    else the attached endpoint's key."""
    return (
        os.environ.get("LLM_API_KEY", "").strip()
        or os.environ.get("MORIE_LLM_API_KEY", "").strip()
        or _cfg.saved_value("own.key")
        or str(_stored_provider().get("api_key", "")).strip()
        or None
    )


def _api_model() -> str:
    """The model for that endpoint: MORIE_API_MODEL, else MORIE_LLM_MODEL, else the saved
    own.model, else the attached endpoint's model, else the default."""
    return (
        os.environ.get("MORIE_API_MODEL", "").strip()
        or os.environ.get("MORIE_LLM_MODEL", "").strip()
        or _cfg.saved_value("own.model")
        or str(_stored_provider().get("api_model", "")).strip()
        or DEFAULT_API_MODEL
    )


def _own_ready() -> bool:
    """Your own endpoint is set (a key is optional: LM Studio, vLLM or llama.cpp need none)."""
    return bool(_api_base_url())


def _openai_key() -> str | None:
    """Return the OPENAI_API_KEY."""
    return os.environ.get("OPENAI_API_KEY", "").strip() or None


def _gemini_key() -> str | None:
    """Return the GEMINI_API_KEY for Google AI Studio."""
    return os.environ.get("GEMINI_API_KEY", "").strip() or None


def _gemini_model() -> str:
    """Return the configured Gemini model name."""
    return os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()


_ollama_cached: bool | None = None


def _hosted_ready() -> bool:
    """True when the user logged in to the hosted tier and its gateway answers."""
    from .hosted import probe_hosted

    return probe_hosted()


def _hosted_attempt(model: str | None) -> tuple[str, str, str | None] | None:
    from .hosted import hosted_base_url, hosted_key, hosted_model_available

    base, key = hosted_base_url(), hosted_key()
    if base and key:
        return (base, model or hosted_model_available(), key)
    return None


def _probe_ollama(timeout: float = _PROBE_TIMEOUT) -> bool:
    """Return True if a local Ollama instance answers AND has a model to use.

    A server with nothing pulled cannot answer a question, so it is not a usable
    route unless a model is named (``OLLAMA_MODEL`` or ``morie config set
    ollama.model``): the automatic order then moves on to your own keys and the
    hosted tier instead of stopping at an Ollama that can only fail.

    The result is cached for the process lifetime to avoid repeated 2-second
    network timeouts on every call to :func:`detect_available_provider`.

    Parameters
    ----------
    timeout : float
        Maximum seconds to wait for the Ollama ``/api/tags`` endpoint.

    Returns
    -------
    bool
        ``True`` when Ollama is reachable and has a model, ``False`` otherwise.
    """
    global _ollama_cached
    if _ollama_cached is not None:
        return _ollama_cached
    tags = _ollama_tags(timeout=timeout)
    _ollama_cached = tags is not None and (bool(tags) or bool(_cfg.value("ollama.model")))
    return _ollama_cached


def _route_name(route: str | None) -> str:
    """``route`` checked, else the saved/env route (``auto`` when nothing says otherwise)."""
    if route is None:
        return _cfg.route()
    r = str(route).strip().lower()
    if r not in _cfg.ROUTES:
        raise ValueError(f"route must be one of: {', '.join(_cfg.ROUTES)}")
    return r


def _hosted_configured() -> bool:
    from .hosted import hosted_base_url, hosted_key

    try:
        return bool(hosted_base_url() and hosted_key())
    except Exception:  # noqa: BLE001 - a services document that cannot be read
        return False


def detect_available_provider(route: str | None = None) -> str:
    """Detect which LLM provider is currently available.

    ``route`` (else ``MORIE_LLM_ROUTE``, else the route saved with ``morie config set route``)
    forces one: ``"ollama"``, ``"hosted"``, or ``"own"`` (your endpoint, else your Gemini or
    OpenAI key). With ``"auto"`` (the default) the order is a local model first, then every key
    of the user's own, then the hosted MORIE tier as a last resort:

    1. **ollama**  -- a local Ollama instance answers and has a model (one pulled,
       or ``OLLAMA_MODEL`` set). A server with nothing pulled is skipped.
    2. **gemini**  -- ``GEMINI_API_KEY`` is set.
    3. **api**     -- your own OpenAI-compatible endpoint is set (``morie config set own.url``,
       ``MORIE_LLM_BASE_URL``/``LLM_API_BASE_URL``, or ``morie provider set``).
    4. **openai**  -- ``OPENAI_API_KEY`` is set.
    5. **hosted**  -- the user holds a MORIE key (issued on request at
       https://rmorie.com/access, or minted by ``morie login``) and the hosted
       gateway, found through the signed services document, answers.
    6. **local**   -- no live provider; MORIE will return static help text.

    Returns
    -------
    str
        One of ``"ollama"``, ``"hosted"``, ``"gemini"``, ``"api"``,
        ``"openai"``, or ``"local"``.

    Examples
    --------
    >>> provider = detect_available_provider()
    >>> provider in ("ollama", "hosted", "gemini", "api", "openai", "local")
    True
    """
    r = _route_name(route)
    if r == "ollama":
        # forced: used even with nothing pulled (the request then says what is missing)
        return _PROVIDER_OLLAMA if _ollama_base_url() else _PROVIDER_LOCAL
    if r == "hosted":
        return _PROVIDER_HOSTED if _hosted_configured() else _PROVIDER_LOCAL
    if r == "own":
        if _own_ready():
            return _PROVIDER_API
        if _gemini_key():
            return _PROVIDER_GEMINI
        if _openai_key():
            return _PROVIDER_OPENAI
        return _PROVIDER_LOCAL

    if _ollama_base_url() and _probe_ollama():
        return _PROVIDER_OLLAMA

    if _gemini_key():
        return _PROVIDER_GEMINI

    if _own_ready():
        return _PROVIDER_API

    if _openai_key():
        return _PROVIDER_OPENAI

    # the hosted tier is a last resort, behind every key of the user's own
    if _hosted_ready():
        return _PROVIDER_HOSTED

    return _PROVIDER_LOCAL


def _provider_model(provider: str) -> str:
    """The model a provider would be asked with (no network beyond the probes already made)."""
    if provider == _PROVIDER_OLLAMA:
        return _ollama_model()
    if provider == _PROVIDER_GEMINI:
        return _gemini_model()
    if provider == _PROVIDER_API:
        return _api_model()
    if provider == _PROVIDER_OPENAI:
        return os.environ.get("MORIE_OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip()
    if provider == _PROVIDER_HOSTED:
        from .hosted import hosted_model_available

        return hosted_model_available()
    return ""


_PROVIDER_LABEL = {
    _PROVIDER_OLLAMA: "Ollama",
    _PROVIDER_GEMINI: "Gemini",
    _PROVIDER_API: "API",
    _PROVIDER_OPENAI: "OpenAI",
    _PROVIDER_HOSTED: "Hosted",
}

_PROVIDER_ROUTE_NAME = {
    _PROVIDER_OLLAMA: "local Ollama",
    _PROVIDER_GEMINI: "your Gemini key",
    _PROVIDER_API: "your own endpoint",
    _PROVIDER_OPENAI: "your OpenAI key",
    _PROVIDER_HOSTED: "hosted MORIE tier",
}


def detect_provider_and_model(route: str | None = None) -> tuple[str, str]:
    """Detect LLM provider and return (provider, human-readable model label).

    Returns
    -------
    tuple[str, str]
        ``(provider_key, display_label)`` -- e.g. ``("gemini", "Gemini:gemini-2.5-flash")``.
    """
    provider = detect_available_provider(route)
    if provider == _PROVIDER_LOCAL:
        return provider, "local fallback (no LLM)"
    return provider, f"{_PROVIDER_LABEL[provider]}:{_provider_model(provider)}"


def detect_model_display() -> dict[str, str]:
    """Return display info with inner (family:size) and outer (model name).

    Returns
    -------
    dict[str, str]
        Keys: ``inner``, ``outer``, ``model``, ``provider``.
        HomeScreen format: ``LLM: {inner} [{outer}]``
    """
    provider = detect_available_provider()
    if provider == _PROVIDER_LOCAL:
        return {"inner": "LOCAL", "outer": "FALLBACK", "model": "", "provider": provider}
    model = _provider_model(provider)
    return {"inner": _PROVIDER_LABEL[provider].upper(), "outer": model.upper(), "model": model, "provider": provider}


def route_status() -> list[dict[str, str]]:
    """One row per route (own endpoint, local Ollama, hosted tier): ``route``, ``status``, ``detail``."""
    rows = []
    own = _api_base_url()
    if own:
        rows.append({"route": "own endpoint", "status": "configured", "detail": f"{own}  model: {_api_model()}"})
    else:
        keys = [n for n, v in (("GEMINI_API_KEY", _gemini_key()), ("OPENAI_API_KEY", _openai_key())) if v]
        rows.append(
            {
                "route": "own endpoint",
                "status": "keys" if keys else "not set",
                "detail": (", ".join(keys) + " set")
                if keys
                else "`morie config set own.url URL` (and own.model, own.key) for any OpenAI-compatible server",
            }
        )
    base = _ollama_base_url()
    if not base:
        rows.append({"route": "local Ollama", "status": "disabled", "detail": "ollama.url = off"})
    else:
        tags = _ollama_tags()
        named = _cfg.value("ollama.model")
        if tags is None:
            rows.append(
                {
                    "route": "local Ollama",
                    "status": "not running",
                    "detail": f"{base} (install Ollama and pull a model, or `morie config set ollama.url ADDRESS`)",
                }
            )
        elif not tags and not named:
            rows.append(
                {
                    "route": "local Ollama",
                    "status": "no models",
                    "detail": f"{base} (`ollama pull NAME`; skipped by the automatic order)",
                }
            )
        else:
            names = ", ".join(t["name"] for t in tags) or "(none listed)"
            rows.append(
                {
                    "route": "local Ollama",
                    "status": "available",
                    "detail": f"{base}  models: {names} (default {_ollama_model()})",
                }
            )
    from . import hosted

    try:
        hbase = hosted.hosted_base_url()
    except Exception as exc:  # noqa: BLE001
        hbase, why = None, str(exc)
    else:
        why = "hosted.url = off or MORIE_HOSTED_BASE_URL empty, or switched off in the services document"
    if not hbase:
        rows.append({"route": "hosted MORIE tier", "status": "disabled", "detail": why})
    elif not hosted.hosted_key():
        rows.append({"route": "hosted MORIE tier", "status": "not logged in", "detail": f"{hbase}  (`morie login`)"})
    elif hosted.probe_hosted():
        listed = hosted.hosted_models() or []
        tail = f"  models: {', '.join(listed)}" if listed else ""
        rows.append(
            {
                "route": "hosted MORIE tier",
                "status": "key stored",
                "detail": f"{hbase}{tail} (default {hosted.hosted_model_available()})",
            }
        )
    else:
        rows.append({"route": "hosted MORIE tier", "status": "key stored", "detail": hosted.hosted_problem_line()})
    return rows


def route_summary(route: str | None = None) -> str:
    """One line on the route ``ask`` takes now, e.g. ``ask uses: hosted MORIE tier, model minimax-m3:cloud``."""
    r = _route_name(route)
    provider = detect_available_provider(r)
    forced = "" if r == "auto" else f"  (route = {r})"
    if provider == _PROVIDER_LOCAL:
        if r == "auto":
            return "ask has no route yet: `morie login` for the hosted tier, or `morie config setup`"
        return f"ask has no route: route = {r} is not set up here (`morie config` shows the settings)"
    model = _provider_model(provider) or "(none)"
    return f"ask uses: {_PROVIDER_ROUTE_NAME[provider]}, model {model}{forced}"


# -- Thinking word synonyms -------------------------------------------------

_THINK_WORDS = [
    "synthesizing",
    "parsing",
    "vectorizing",
    "optimizing",
    "brewing",
    "ruminating",
    "pondering",
    "wrangling pixels",
    "consulting the scrolls",
    "envisioning",
    "distilling",
    "weaving",
    "crystallizing",
    "tracing",
    "fluxing",
    "modulating",
    "sequencing",
    "combobulating",
    "calibrating",
    "interpolating",
    "decomposing",
    "iterating",
    "compiling gradients",
    "tuning hyperparameters",
    "aligning embeddings",
]

_CONTEXT_WORDS: dict[str, str] = {
    "monte carlo": "simulating Monte Carlo",
    "markov": "simulating Markov Chains",
    "counterfactual": "estimating counterfactuals",
    "propensity": "scoring propensities",
    "bootstrap": "bootstrapping",
    "regression": "fitting regression surfaces",
    "bayesian": "sampling posteriors",
    "causal": "tracing causal paths",
    "survival": "modeling survival curves",
    "genomic": "sequencing loci",
    "epigenetic": "mapping methylation",
    "sample": "drawing samples",
    "hypothesis": "testing hypotheses",
    "dml": "cross-fitting folds",
    "forest": "growing random forests",
    "neural": "propagating activations",
    "cluster": "partitioning clusters",
    "pca": "reducing dimensions",
    "variance": "decomposing variance",
    "likelihood": "maximizing likelihood",
    "posterior": "sampling posteriors",
    "prior": "eliciting priors",
    "iv": "instrumenting variables",
    "matching": "pairing counterfactuals",
    "weight": "calibrating weights",
    "treatment": "estimating treatment effects",
    "power": "computing power curves",
    "odds": "computing odds ratios",
    "hazard": "modeling hazard rates",
    "genome": "scanning the genome",
    "methylation": "mapping CpG islands",
    "gene": "annotating gene variants",
    "protein": "folding protein structures",
    "cell": "profiling cell types",
    "drug": "screening compounds",
    "trial": "designing trial arms",
    "randomiz": "allocating treatment arms",
    "stratif": "stratifying strata",
    "confound": "adjusting for confounders",
    "bias": "diagnosing bias sources",
    "missing": "imputing missing values",
    "outlier": "flagging outliers",
    "time series": "forecasting trajectories",
    "spatial": "mapping spatial fields",
    "network": "tracing network edges",
    "graph": "traversing graph paths",
    "entropy": "measuring information entropy",
    "game theory": "solving equilibria",
    "nash": "finding Nash equilibria",
    "mechanism": "designing mechanisms",
    "auction": "simulating auctions",
}


def pick_thinking_word(query: str) -> str:
    """Pick a context-aware thinking word based on the query, or a random one."""
    import random

    q = query.lower()
    # Check context keywords first
    for kw, phrase in _CONTEXT_WORDS.items():
        if kw in q:
            return phrase
    return random.choice(_THINK_WORDS)


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------


def build_morie_context(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Build an LLM-friendly context dictionary from the MORIE package state.

    The returned dictionary is designed to be injected into the system prompt
    so the LLM is aware of the available modules, the CPADS data contract,
    and the current working directory.

    Parameters
    ----------
    repo_root : str | Path | None
        Path to the MORIE repository root.  When ``None`` the function
        attempts to resolve the root from this file's location.

    Returns
    -------
    dict[str, Any]
        A dictionary with keys:

        - ``module_list`` -- list of module name/description pairs.
        - ``cpads_schema`` -- the CPADS data contract dictionary.
        - ``cwd`` -- the current working directory as a string.
        - ``repo_root`` -- the resolved repository root, or ``"unknown"``.

    Examples
    --------
    >>> ctx = build_morie_context()
    >>> "module_list" in ctx and "cpads_schema" in ctx
    True
    """
    if repo_root is None:
        # Attempt to resolve from file location: llm.py -> morie/ -> py-package/ -> morie-root/
        try:
            repo_root = str(Path(__file__).resolve().parents[2])
        except Exception:
            repo_root = "unknown"
    else:
        repo_root = str(Path(repo_root).resolve())

    module_list = [{"name": spec.name, "description": spec.description} for spec in MODULE_SPECS.values()]

    return {
        "module_list": module_list,
        "cpads_schema": cpads_contract(),
        "cwd": os.getcwd(),
        "repo_root": repo_root,
        "function_signatures": _collect_function_signatures(),
        "dataset_schema": _current_dataset_schema(),
        "stat_commands": _stat_command_summary(),
    }


_MORIE_MODULES = [
    "morie.quant",
    "morie.causal",
    "morie.effects",
    "morie.survey",
    "morie.inference",
    "morie.did",
    "morie.rdd",
    "morie.iv",
    "morie.matching",
    "morie.survival",
    "morie.sensitivity",
    "morie.ml",
    "morie.ebac",
    "morie.sampling",
    "morie.loc",
    "morie.emissions",
    "morie.data",
    "morie.modules",
]


def get_last_traceback() -> str:
    """Return the last Python traceback, if any, for error-context injection."""
    import sys
    import traceback

    exc = sys.last_value if hasattr(sys, "last_value") else None
    if exc is None:
        return ""
    tb = getattr(sys, "last_traceback", None)
    if tb is None:
        return f"{type(exc).__name__}: {exc}"
    lines = traceback.format_exception(type(exc), exc, tb)
    text = "".join(lines)
    return text[-1000:] if len(text) > 1000 else text


def _retrieve_relevant_source(query: str, max_chars: int = 1500) -> str:
    """RAG: retrieve actual source code relevant to the user's query.

    Searches morie module functions for keyword matches against the query,
    then returns the full docstring + signature of matching functions.
    This gives the LLM actual code to reference instead of hallucinating.

    Parameters
    ----------
    query : str
        The user's question.
    max_chars : int
        Max characters of source to inject.

    Returns
    -------
    str
        Relevant source code snippets, or empty string.
    """
    import importlib
    import inspect
    import re

    # Extract keywords from query (lowercase, strip punctuation)
    keywords = set(re.findall(r"[a-z_]{3,}", query.lower()))
    # Add common synonyms
    if "ipw" in keywords:
        keywords.update({"propensity", "weight", "inverse"})
    if "ate" in keywords:
        keywords.update({"treatment", "effect", "estimate"})
    if "quant" in keywords or "turboquant" in keywords:
        keywords.update({"quantize", "codebook", "rotation", "turboquant"})
    if "qjl" in keywords:
        keywords.update({"sign", "projection", "residual", "encode", "decode"})
    if "load" in keywords or "cpads" in keywords or "dataset" in keywords:
        keywords.update({"load_dataset", "dataset", "cpads", "load"})
    if "dml" in keywords:
        keywords.update({"double", "machine", "plr", "estimate"})

    matches: list[tuple[float, str]] = []

    for mod_name in _MORIE_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue

        for name in dir(mod):
            if name.startswith("_"):
                continue
            obj = getattr(mod, name, None)
            if obj is None or not callable(obj):
                continue
            obj_mod = getattr(obj, "__module__", "")
            if not obj_mod.startswith("morie"):
                continue

            # Score by keyword overlap with function name + docstring
            fn_lower = name.lower()
            doc = (getattr(obj, "__doc__", "") or "").lower()
            score = 0.0
            for kw in keywords:
                if kw in fn_lower:
                    score += 3.0  # strong match on function name
                if kw in doc[:200]:
                    score += 1.0  # match in docstring

            if score > 0:
                try:
                    sig = inspect.signature(obj)
                    full_doc = getattr(obj, "__doc__", "") or ""
                    snippet = f"{mod_name.split('.')[-1]}.{name}{sig}\n{full_doc}"
                    matches.append((score, snippet))
                except (ValueError, TypeError):
                    pass

    if not matches:
        return ""

    # Sort by relevance score, take top matches
    matches.sort(key=lambda x: x[0], reverse=True)
    result_parts: list[str] = []
    total = 0
    for _score, snippet in matches:
        if total + len(snippet) > max_chars:
            break
        result_parts.append(snippet)
        total += len(snippet)

    return "\n---\n".join(result_parts) if result_parts else ""


def _collect_function_signatures() -> list[dict[str, str]]:
    """Introspect key morie modules for function names + one-line descriptions."""
    import importlib

    sigs: list[dict[str, str]] = []
    for mod_name in _MORIE_MODULES[:14]:
        try:
            mod = importlib.import_module(mod_name)
            for name in sorted(dir(mod)):
                if name.startswith("_"):
                    continue
                obj = getattr(mod, name, None)
                if obj is None or not callable(obj):
                    continue
                if not hasattr(obj, "__doc__") or not obj.__doc__:
                    continue
                # Skip type aliases, dataclass decorators, stdlib re-exports
                obj_mod = getattr(obj, "__module__", "")
                if not obj_mod.startswith("morie"):
                    continue
                first_line = obj.__doc__.strip().split("\n")[0][:80]
                sigs.append({"fn": f"{mod_name.split('.')[-1]}.{name}", "desc": first_line})
        except ImportError:
            continue
    return sigs[:60]


def _current_dataset_schema() -> dict[str, str] | None:
    """If a DataFrame is loaded in MORIEApp, return its column schema."""
    try:
        from . import tui as _tui_mod

        app = getattr(_tui_mod, "_running_app", None)
        if app and hasattr(app, "loaded_df") and app.loaded_df is not None:
            df = app.loaded_df
            return {col: str(dtype) for col, dtype in df.dtypes.items()}
    except Exception:
        pass
    return None


def _stat_command_summary() -> list[str]:
    """Return top stat command names grouped concisely."""
    try:
        from .stat_commands import all_command_names

        return all_command_names()[:100]
    except Exception:
        try:
            from .stat_commands import COMMAND_REGISTRY

            return sorted(COMMAND_REGISTRY.keys())[:100]
        except Exception:
            return []


def _format_context_block(context: dict[str, Any] | None) -> str:
    """Render a context dictionary into a text block for the system prompt.

    Caps total output at ~2000 chars to avoid blowing the context window.
    """
    if not context:
        return ""

    parts: list[str] = []

    modules = context.get("module_list")
    if modules:
        names = ", ".join(m["name"] for m in modules)
        parts.append(f"Available MORIE modules: {names}")

    schema = context.get("cpads_schema")
    if schema:
        req_vars = schema.get("required_variables", [])
        parts.append(f"CPADS required variables: {', '.join(req_vars)}")

    # Dataset schema (if a DataFrame is loaded)
    ds_schema = context.get("dataset_schema")
    if ds_schema:
        cols = [f"{c}({t})" for c, t in list(ds_schema.items())[:20]]
        parts.append(f"Loaded dataset columns: {', '.join(cols)}")

    # Function signatures (top ones)
    fn_sigs = context.get("function_signatures")
    if fn_sigs:
        sig_lines = [f"{s['fn']}: {s['desc']}" for s in fn_sigs[:25]]
        parts.append("Key functions:\n" + "\n".join(sig_lines))

    # Stat commands
    stat_cmds = context.get("stat_commands")
    if stat_cmds:
        parts.append(f"Stat commands ({len(stat_cmds)}): {', '.join(stat_cmds[:40])}")

    cwd = context.get("cwd")
    if cwd:
        parts.append(f"User working directory: {cwd}")

    # RAG: inject relevant source code for the current query
    rag = context.get("rag_source")
    if rag:
        parts.append(f"RELEVANT SOURCE CODE (use this to answer accurately):\n{rag}")

    block = "\n".join(parts)
    if len(block) > 3500:
        block = block[:3500] + "\n..."
    return block


# ---------------------------------------------------------------------------
# Chat completions helpers
# ---------------------------------------------------------------------------


def _build_messages(
    prompt: str,
    context: dict[str, Any] | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Build the ``messages`` array for the chat completions payload."""
    if system_prompt is None:
        context_block = _format_context_block(context)
        system_prompt = _MORIE_SYSTEM_PROMPT_TEMPLATE.format(context_block=context_block)
    elif context and context.get("module_list"):
        mods = "\n".join(f"- {m['name']}: {m['description']}" for m in context["module_list"])
        system_prompt = (
            f"{system_prompt}\n\nmorie's analysis modules (`morie run-module NAME`); recommend only these, "
            f"and say so when none fits:\n{mods}"
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def _chat_url(base_url: str) -> str:
    """The chat-completions URL for a base: ``BASE/chat/completions`` when the base already ends in
    an API version (``.../v1``, Gemini's ``.../v1beta/openai``), else ``BASE/v1/chat/completions``."""
    base = base_url.rstrip("/")
    if re.search(r"/v\d+[a-z0-9]*(/openai)?$", base, re.I):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _request_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str | None = None,
    stream: bool = False,
    timeout: float = _REQUEST_TIMEOUT,
    max_tokens: int = 4096,
) -> httpx.Response:
    """Send a POST to ``/v1/chat/completions`` and return the raw response.

    Parameters
    ----------
    base_url : str
        The provider base URL (e.g., ``http://localhost:11434`` for Ollama,
        ``https://api.openai.com`` for OpenAI).
    model : str
        The model identifier to use.
    messages : list[dict[str, str]]
        The chat messages array.
    api_key : str | None
        Bearer token.  Omitted for local Ollama requests.
    stream : bool
        Whether to request server-sent-event streaming.
    timeout : float
        Request timeout in seconds.

    Returns
    -------
    httpx.Response
        The raw ``httpx`` response object.
    """
    url = _chat_url(base_url)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }

    # reasoning models (the hosted default) spend tokens thinking before answering: give them room,
    # or the answer comes back as an empty "content" with the budget gone
    payload["max_tokens"] = max_tokens
    if "localhost" in base_url or "127.0.0.1" in base_url:
        timeout = max(timeout, 300.0)

    return httpx.post(
        url,
        json=payload,
        headers=headers,
        timeout=timeout,
    )


def _extract_text(response: httpx.Response) -> str:
    """Extract the assistant message text from a non-streaming response."""
    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        logger.warning("LLM response contained no choices: %s", data)
        return ""
    return choices[0].get("message", {}).get("content", "") or ""


class EmptyAnswerError(Exception):
    """The provider answered with no text, twice: the chain moves on to the next provider."""


def _completion_text(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str | None,
    timeout: float,
) -> str:
    """One non-streaming completion, with one retry when the answer comes back empty.

    A thinking model sometimes spends its whole budget on reasoning and returns
    an empty ``content``; the retry gives it four times the room. An answer that
    is still empty is reported as :class:`EmptyAnswerError` rather than returned as
    silence, so the caller's provider chain can try the next provider.
    """
    for budget in (4096, 16384):
        resp = _request_completion(
            base_url, model, messages, api_key=api_key, stream=False, timeout=timeout, max_tokens=budget
        )
        resp.raise_for_status()
        text = _extract_text(resp)
        if text.strip():
            return text
        logger.warning("Empty answer from %s (%s) with max_tokens=%d; retrying", base_url, model, budget)
    raise EmptyAnswerError(f"{model} at {base_url} answered with no text")


def _iter_stream(response: httpx.Response) -> Iterator[str]:
    """Yield text chunks from a non-streaming SSE response already in memory.

    This helper is kept for backward compatibility; prefer
    :func:`_stream_completion` for live streaming over an open connection.

    Yields
    ------
    str
        Each content delta string as it arrives.
    """
    for line in response.text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line == "data: [DONE]":
            break
        if line.startswith("data: "):
            raw = line[len("data: ") :]
            try:
                chunk = json.loads(raw)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
            except (json.JSONDecodeError, IndexError, KeyError):
                continue


def _stream_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str | None = None,
    timeout: float = _REQUEST_TIMEOUT,
) -> Iterator[str]:
    """Stream a chat completion, keeping the HTTP connection open.

    Uses ``httpx.stream()`` so the connection stays open while the generator
    is being consumed.  The connection is closed automatically when the
    generator is exhausted or garbage-collected.

    Parameters
    ----------
    base_url : str
        The provider base URL.
    model : str
        Model identifier.
    messages : list[dict[str, str]]
        Chat messages array.
    api_key : str | None
        Bearer token, or ``None`` for local Ollama.
    timeout : float
        Request timeout in seconds.

    Yields
    ------
    str
        Each content delta as it arrives from the SSE stream.
    """
    url = _chat_url(base_url)
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    # reasoning models (the hosted default) spend tokens thinking before answering: give them room,
    # or the answer comes back as an empty "content" with the budget gone
    payload["max_tokens"] = 4096
    if "localhost" in base_url or "127.0.0.1" in base_url:
        timeout = max(timeout, 300.0)

    with httpx.stream("POST", url, json=payload, headers=headers, timeout=timeout) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            line = line.strip()
            if not line:
                continue
            if line == "data: [DONE]":
                return
            if line.startswith("data: "):
                raw = line[len("data: ") :]
                try:
                    chunk = json.loads(raw)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue


# ---------------------------------------------------------------------------
# Local fallback
# ---------------------------------------------------------------------------

_LOCAL_FALLBACK_TEXT = """\
MORIE is running in local-only mode (no LLM provider detected).

The analyses do not need a model:
  - morie list-modules          List the analysis modules
  - morie run-module <name>     Run one module
  - morie pipeline --all -y     Run the full analysis pipeline

To get answers from a model, enable one of these (tried in this order):

  1. A local Ollama (private, no key):
       curl -fsSL https://ollama.com/install.sh | sh
       ollama pull gemma4:e2b

  2. The hosted MORIE tier at https://llm.rmorie.com (one key per user,
     rate-limited, nothing you send is stored):
       morie login                          # GitHub device flow
       morie login --email you@example.com  # a code sent to your inbox
       morie login --token                  # paste a key from the website

  3. Your own key:
       export GEMINI_API_KEY="..."                              # Gemini
       export LLM_API_BASE_URL="..." LLM_API_KEY="..."          # any OpenAI-compatible endpoint
       export OPENAI_API_KEY="..."                              # OpenAI

`morie doctor` reports what is reachable from here.
"""


class _FallbackText(str):
    """The static local-mode answer. A str subclass so callers can tell
    "no backend answered" apart from a live answer with the same words."""


def _local_fallback(prompt: str) -> str:
    """Return a helpful local response when no LLM provider is available.

    Parameters
    ----------
    prompt : str
        The user's original question (used for keyword matching).

    Returns
    -------
    str
        A static help message with package usage guidance.
    """
    normalized = prompt.lower()
    sections = [_LOCAL_FALLBACK_TEXT.strip()]

    # Provide topic-specific hints based on keyword matching.
    if "cpads" in normalized or "dataset" in normalized or "data" in normalized:
        contract = cpads_contract()
        sections.append(
            "CPADS data contract:\n"
            f"  Required variables: {', '.join(contract['required_variables'])}\n"
            f"  Expected path: {contract['expected_wrangled_path']}"
        )

    if "ipw" in normalized or "propensity" in normalized or "causal" in normalized:
        sections.append(
            "Causal inference modules available: propensity-scores, "
            "causal-estimators, treatment-effects, ebac-selection-adjustment-ipw.\n"
            "Use `morie run-module <name>` to execute."
        )

    if "module" in normalized or "list" in normalized:
        names = [spec.name for spec in MODULE_SPECS.values()]
        sections.append("Implemented modules: " + ", ".join(names))

    return _FallbackText("\n\n".join(sections))


# ---------------------------------------------------------------------------
# Main ask() function
# ---------------------------------------------------------------------------


class ModelNotOnKeyError(RuntimeError):
    """The hosted gateway refused a model the caller named: it is not on this key."""


def _model_not_on_key(exc: Exception, base_url: str, model: str | None) -> bool:
    """A named model the hosted gateway answers 403/404 for: say so rather than try the next provider."""
    if not model or not isinstance(exc, httpx.HTTPStatusError):
        return False
    hosted = _hosted_attempt(model)
    return bool(hosted) and base_url == hosted[0] and exc.response.status_code in (403, 404)


def _tagged_attempts(provider: str, model: str | None) -> list[tuple[str, str, str, str | None]]:
    """The ordered (route, base_url, model, api_key) attempts for a provider.

    ``route`` is ``ollama``, ``own`` (your endpoint or keys) or ``hosted``. An attempt with no
    model to ask (an Ollama with nothing pulled and no ``OLLAMA_MODEL``) is left out: it could
    only fail, and with streaming its failure would surface after the chain had already chosen it.
    """
    attempts: list[tuple[str, str, str, str | None]] = []
    own_chain = {
        _PROVIDER_OLLAMA: (_PROVIDER_GEMINI, _PROVIDER_API, _PROVIDER_OPENAI),
        _PROVIDER_GEMINI: (_PROVIDER_GEMINI, _PROVIDER_API, _PROVIDER_OPENAI),
        _PROVIDER_API: (_PROVIDER_API, _PROVIDER_OPENAI),
        _PROVIDER_OPENAI: (_PROVIDER_OPENAI,),
    }.get(provider, ())
    if provider == _PROVIDER_OLLAMA and _ollama_base_url():
        attempts.append(("ollama", _ollama_base_url(), model or _ollama_model(), _ollama_key()))  # type: ignore[arg-type]
    # the user's own keys, then the hosted tier, if the first choice fails at request time
    for p in own_chain:
        if p == _PROVIDER_GEMINI and _gemini_key():
            attempts.append(("own", GEMINI_BASE_URL, model or _gemini_model(), _gemini_key()))
        elif p == _PROVIDER_API and _own_ready():
            attempts.append(("own", _api_base_url(), model or _api_model(), _api_key()))  # type: ignore[arg-type]
        elif p == _PROVIDER_OPENAI and _openai_key():
            attempts.append(("own", OPENAI_BASE_URL, model or DEFAULT_OPENAI_MODEL, _openai_key()))
    if provider in (_PROVIDER_OLLAMA, _PROVIDER_HOSTED) or own_chain:
        hosted = _hosted_attempt(model)
        if hosted:
            attempts.append(("hosted", *hosted))
    return [a for a in attempts if a[2]]


def _provider_attempts(provider: str, model: str | None) -> list[tuple[str, str, str | None]]:
    """The ordered (base_url, model, api_key) attempts for a provider, the same for ask() and ask_multi()."""
    return [a[1:] for a in _tagged_attempts(provider, model)]


def _attempts_for(provider: str, model: str | None, route: str) -> list[tuple[str, str, str | None]]:
    """The attempts, kept to the forced route when one is set (no falling back to another route)."""
    tagged = _tagged_attempts(provider, model)
    if route != "auto":
        tagged = [a for a in tagged if a[0] == route]
    return [a[1:] for a in tagged]


def _run_attempts(
    attempts: list[tuple[str, str, str | None]],
    messages: list[dict[str, str]],
    *,
    stream: bool,
    model: str | None,
    timeout: float,
) -> str | Iterator[str] | None:
    """Try each attempt in order; the first answer wins, None when every one failed.

    A streamed answer is primed with its first chunk here, so a provider that fails
    (a refused model, a 4xx/5xx, a dead connection) falls through to the next one
    instead of failing in the caller after the chain has already been left.
    """
    last_error: Exception | None = None
    for base_url, req_model, api_key in attempts:
        try:
            logger.debug("Attempting LLM request: base_url=%s model=%s stream=%s", base_url, req_model, stream)
            if stream:
                # _stream_completion keeps the httpx connection open while the generator is live
                gen = iter(_stream_completion(base_url, req_model, messages, api_key=api_key, timeout=timeout))
                first = next(gen, None)
                if first is None:
                    raise EmptyAnswerError(f"{req_model} at {base_url} streamed no text")
                return itertools.chain([first], gen)
            return _completion_text(base_url, req_model, messages, api_key=api_key, timeout=timeout)
        except (httpx.HTTPError, httpx.TimeoutException, OSError, KeyError, EmptyAnswerError) as exc:
            if _model_not_on_key(exc, base_url, model):
                raise ModelNotOnKeyError(
                    f"model {model!r} is not available on your hosted key; `morie models` lists the ones it can use"
                ) from None
            last_error = exc
            logger.warning("Provider at %s failed: %s. Trying next provider.", base_url, exc)
    if last_error is not None:
        logger.warning("All LLM providers failed. Last error: %s. Falling back to local mode.", last_error)
    return None


def ask(
    prompt: str,
    context: dict[str, Any] | None = None,
    *,
    stream: bool = False,
    model: str | None = None,
    provider: str | None = None,
    route: str | None = None,
    system_prompt: str | None = None,
    timeout: float = _REQUEST_TIMEOUT,
) -> str | Iterator[str]:
    """Send a prompt to the best available LLM provider and return the response.

    The provider chain is: Ollama (local, when it has a model) -> your own keys and
    OpenAI-compatible endpoint -> the hosted MORIE tier -> local fallback. Each provider
    is tried in order; on failure the next is attempted.

    Parameters
    ----------
    prompt : str
        The user's question or instruction.
    context : dict[str, Any] | None
        Optional context dictionary (e.g., from :func:`build_morie_context`).
        Injected into the system prompt to give the LLM awareness of available
        modules, CPADS schema, and the user's working directory.
    stream : bool
        If ``True``, return an iterator of string chunks for streaming output.
        If ``False`` (default), return the full response as a single string.
    model : str | None
        Override the model identifier.  When ``None``, a sensible default is
        chosen per provider.
    provider : str | None
        Force a specific provider (``"ollama"``, ``"api"``, ``"openai"``,
        ``"local"``).  When ``None``, :func:`detect_available_provider` is
        used to auto-detect.
    route : str | None
        ``"auto"``, ``"own"``, ``"ollama"`` or ``"hosted"``. When ``None``, the
        route saved with ``morie config set route`` (or ``MORIE_LLM_ROUTE``) applies,
        else ``"auto"``. A forced route never falls back to another one.
    system_prompt : str | None
        Override the entire system prompt.  When ``None``, the standard MORIE
        system prompt is built from the ``context`` parameter.
    timeout : float
        HTTP request timeout in seconds.

    Returns
    -------
    str | Iterator[str]
        The LLM response text (or a streaming iterator of text chunks).
        When all providers fail, returns a local fallback help string.

    Examples
    --------
    Needs a provider (a stored MORIE key, a local Ollama, or a provider key):

    >>> # Non-streaming (returns full text)
    >>> response = ask("What is AIPW?")  # doctest: +SKIP
    >>> isinstance(response, str)  # doctest: +SKIP
    True

    >>> # Streaming: chunks print as they arrive
    >>> for chunk in ask("Explain TMLE", stream=True):  # doctest: +SKIP
    ...     print(chunk, end="")
    """
    r = _route_name(route)
    if provider is None:
        provider = detect_available_provider() if route is None else detect_available_provider(r)

    if provider == _PROVIDER_LOCAL:
        result = _local_fallback(prompt)
        if stream:
            return iter([result])
        return result

    # RAG: retrieve relevant source code for the query
    rag_source = _retrieve_relevant_source(prompt)
    if rag_source:
        if context is None:
            context = {}
        context["rag_source"] = rag_source

    # Inject last traceback if user seems to be debugging
    tb = get_last_traceback()
    if tb and any(
        kw in prompt.lower() for kw in ("error", "bug", "fix", "traceback", "fail", "broke", "crash", "debug")
    ):
        if context is None:
            context = {}
        context["rag_source"] = (context.get("rag_source", "") + f"\n\nLAST ERROR:\n{tb}").strip()

    messages = _build_messages(prompt, context=context, system_prompt=system_prompt)

    attempts = _attempts_for(provider, model, r)
    out = _run_attempts(attempts, messages, stream=stream, model=model, timeout=timeout) if attempts else None
    if out is not None:
        return out
    result = _local_fallback(prompt)
    return iter([result]) if stream else result


# ---------------------------------------------------------------------------
# Multi-turn conversation support
# ---------------------------------------------------------------------------


def ask_multi(
    messages: list[dict[str, str]],
    *,
    stream: bool = False,
    model: str | None = None,
    provider: str | None = None,
    route: str | None = None,
    timeout: float = _REQUEST_TIMEOUT,
) -> str | Iterator[str]:
    """Send a pre-built messages array to the best available LLM provider.

    Unlike :func:`ask`, this accepts the full ``messages`` array directly,
    enabling multi-turn conversation support.  The caller is responsible for
    constructing the system and user messages.

    Parameters
    ----------
    messages : list[dict[str, str]]
        The chat messages array (system, user, assistant turns).
    stream : bool
        If ``True``, return an iterator of string chunks.
    model : str | None
        Override the model identifier.
    provider : str | None
        Force a specific provider.  Auto-detected when ``None``.
    route : str | None
        ``"auto"``, ``"own"``, ``"ollama"`` or ``"hosted"``; ``None`` uses the saved route.
    timeout : float
        HTTP request timeout in seconds.

    Returns
    -------
    str | Iterator[str]
        The LLM response text (or a streaming iterator).
    """
    r = _route_name(route)
    if provider is None:
        provider = detect_available_provider() if route is None else detect_available_provider(r)

    def fallback() -> str | Iterator[str]:
        user_msgs = [m for m in messages if m.get("role") == "user"]
        result = _local_fallback(user_msgs[-1]["content"] if user_msgs else "")
        return iter([result]) if stream else result

    if provider == _PROVIDER_LOCAL:
        return fallback()
    attempts = _attempts_for(provider, model, r)
    out = _run_attempts(attempts, messages, stream=stream, model=model, timeout=timeout) if attempts else None
    return fallback() if out is None else out


# ---------------------------------------------------------------------------
# Agent availability check
# ---------------------------------------------------------------------------


def agent_available() -> bool:
    """Return True when at least one live LLM provider is available.

    Returns
    -------
    bool
        ``True`` if a live provider is detected, ``False`` if only local
        fallback is available.

    Examples
    --------
    >>> isinstance(agent_available(), bool)
    True
    """
    return detect_available_provider() != _PROVIDER_LOCAL


assistant_available = agent_available
