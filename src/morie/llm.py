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

Provider priority (auto-detected at runtime):
    1. Ollama    -- local, private, no API key needed
    2. Gemini    -- Google AI, generous free tier
    3. API       -- generic OpenAI-compatible (Qwen, GPT-OSS, Groq, etc.)
    4. OpenAI    -- official OpenAI API
    5. local     -- static help text, no network required

References
----------
* Ollama API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
* Gemini OpenAI-compatible API: https://ai.google.dev/gemini-api/docs/openai
* OpenAI Chat Completions API: https://platform.openai.com/docs/api-reference/chat
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

from .cpads import cpads_contract
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


def _ollama_base_url() -> str:
    """Return the configured Ollama base URL."""
    return os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")


_ollama_model_cached: str | None = None


def _ollama_model() -> str:
    """Return the Ollama model to use -- auto-detected from the running instance.

    Priority:
    1. MORIE_OLLAMA_MODEL env var (explicit override)
    2. First model from ``ollama list`` (auto-detect)
    3. Empty string (no model available)

    Cached for the process lifetime after first detection.
    """
    global _ollama_model_cached
    if _ollama_model_cached is not None:
        return _ollama_model_cached

    # 1. Env var override
    env = os.environ.get("MORIE_OLLAMA_MODEL", "").strip()
    if env:
        _ollama_model_cached = env
        return env

    # 2. Auto-detect from running Ollama -- prefer largest perseus:* model
    try:
        from .loc import LocalOllama

        client = LocalOllama()
        models = client.list_models()
        if models:
            perseus_models = [m for m in models if m.name.startswith("perseus")]
            if perseus_models:
                perseus_models.sort(key=lambda m: m.size, reverse=True)
                _ollama_model_cached = perseus_models[0].name
                return perseus_models[0].name
            _ollama_model_cached = models[0].name
            return models[0].name
    except Exception:
        pass

    _ollama_model_cached = DEFAULT_OLLAMA_MODEL
    return DEFAULT_OLLAMA_MODEL


def _stored_provider() -> dict:
    """The endpoint attached with `morie provider set` (shared credentials file)."""
    try:
        from .hosted import read_credentials

        data = read_credentials()
    except Exception:
        return {}
    return {k: v for k, v in data.items() if k in ("api_base_url", "api_key", "api_model") and v}


def _api_base_url() -> str | None:
    """The generic OpenAI-compatible base URL: LLM_API_BASE_URL, else the attached endpoint."""
    url = os.environ.get("LLM_API_BASE_URL", "").strip() or str(_stored_provider().get("api_base_url", "")).strip()
    return url.rstrip("/") if url else None


def _api_key() -> str | None:
    """The key for that endpoint: LLM_API_KEY, else the attached endpoint's key."""
    return os.environ.get("LLM_API_KEY", "").strip() or str(_stored_provider().get("api_key", "")).strip() or None


def _api_model() -> str:
    """The model for that endpoint: MORIE_API_MODEL, else the attached endpoint's model, else the default."""
    return (
        os.environ.get("MORIE_API_MODEL", "").strip()
        or str(_stored_provider().get("api_model", "")).strip()
        or DEFAULT_API_MODEL
    )


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
    """Return True if a local Ollama instance responds to a health check.

    The result is cached for the process lifetime to avoid repeated 2-second
    network timeouts on every call to :func:`detect_available_provider`.

    Uses :class:`morie.loc.LocalOllama` for the probe.

    Parameters
    ----------
    timeout : float
        Maximum seconds to wait for the Ollama ``/api/tags`` endpoint.

    Returns
    -------
    bool
        ``True`` when Ollama is reachable, ``False`` otherwise.
    """
    global _ollama_cached
    if _ollama_cached is not None:
        return _ollama_cached
    try:
        from .loc import LocalOllama

        client = LocalOllama(base_url=_ollama_base_url())
        _ollama_cached = client.is_running(timeout=timeout)
    except Exception:
        _ollama_cached = False
    return _ollama_cached


def detect_available_provider() -> str:
    """Detect which LLM provider is currently available.

    The detection order mirrors the provider chain priority:

    1. **ollama**  -- a local Ollama instance is reachable (probed via HTTP).
    2. **hosted**  -- the user ran ``morie login`` and llm.rmorie.com answers.
    3. **gemini**  -- ``GEMINI_API_KEY`` is set.
    4. **api**     -- ``LLM_API_BASE_URL`` and ``LLM_API_KEY`` are set.
    5. **openai**  -- ``OPENAI_API_KEY`` is set.
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
    if _probe_ollama():
        return _PROVIDER_OLLAMA

    if _hosted_ready():
        return _PROVIDER_HOSTED

    if _gemini_key():
        return _PROVIDER_GEMINI

    if _api_base_url() and _api_key():
        return _PROVIDER_API

    if _openai_key():
        return _PROVIDER_OPENAI

    return _PROVIDER_LOCAL


def detect_provider_and_model() -> tuple[str, str]:
    """Detect LLM provider and return (provider, human-readable model label).

    Returns
    -------
    tuple[str, str]
        ``(provider_key, display_label)`` -- e.g. ``("gemini", "Gemini:gemini-2.5-flash")``.
    """
    provider = detect_available_provider()

    if provider == _PROVIDER_OLLAMA:
        model = _ollama_model()
        return provider, f"Ollama:{model}"

    if provider == _PROVIDER_GEMINI:
        model = os.environ.get("MORIE_GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
        return provider, f"Gemini:{model}"

    if provider == _PROVIDER_API:
        return provider, f"API:{_api_model()}"

    if provider == _PROVIDER_OPENAI:
        model = os.environ.get("MORIE_OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip()
        return provider, f"OpenAI:{model}"

    return provider, "local fallback (no LLM)"


def detect_model_display() -> dict[str, str]:
    """Return display info with inner (family:size) and outer (model name).

    Returns
    -------
    dict[str, str]
        Keys: ``inner``, ``outer``, ``model``, ``provider``.
        HomeScreen format: ``LLM: {inner} [{outer}]``
    """
    provider = detect_available_provider()
    if provider == _PROVIDER_OLLAMA:
        model = _ollama_model()
        return {"inner": "OLLAMA", "outer": model.upper(), "model": model, "provider": provider}
    if provider == _PROVIDER_GEMINI:
        model = os.environ.get("MORIE_GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
        return {"inner": "GEMINI", "outer": model.upper(), "model": model, "provider": provider}
    return {"inner": "LOCAL", "outer": "FALLBACK", "model": "", "provider": provider}


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
        from .stat_commands import list_all_commands

        return list_all_commands()[:100]
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

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


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
    url = f"{base_url}/v1/chat/completions"

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
    url = f"{base_url}/v1/chat/completions"
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


def ask(
    prompt: str,
    context: dict[str, Any] | None = None,
    *,
    stream: bool = False,
    model: str | None = None,
    provider: str | None = None,
    system_prompt: str | None = None,
    timeout: float = _REQUEST_TIMEOUT,
) -> str | Iterator[str]:
    """Send a prompt to the best available LLM provider and return the response.

    The provider chain is: Ollama (local) -> OpenAI-compatible API -> OpenAI
    direct -> local fallback.  Each provider is tried in order; on failure the
    next is attempted.

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
    >>> # Non-streaming (returns full text)
    >>> response = ask("What is AIPW?")
    >>> isinstance(response, str)
    True

    >>> # Streaming
    >>> for chunk in ask("Explain TMLE", stream=True):
    ...     print(chunk, end="")
    """
    if provider is None:
        provider = detect_available_provider()

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

    # Build the ordered list of (base_url, model, api_key) to try.
    attempts: list[tuple[str, str, str | None]] = []

    if provider == _PROVIDER_OLLAMA:
        attempts.append(
            (
                _ollama_base_url(),
                model or _ollama_model(),
                None,
            )
        )
        # Fallback chain if Ollama fails at request time.
        if _hosted_attempt(model):
            attempts.append(_hosted_attempt(model))  # type: ignore[arg-type]
        if _gemini_key():
            attempts.append((GEMINI_BASE_URL, model or _gemini_model(), _gemini_key()))
        if _api_base_url() and _api_key():
            attempts.append((_api_base_url(), model or _api_model(), _api_key()))  # type: ignore[arg-type]
        if _openai_key():
            attempts.append((OPENAI_BASE_URL, model or DEFAULT_OPENAI_MODEL, _openai_key()))

    elif provider == _PROVIDER_HOSTED:
        if _hosted_attempt(model):
            attempts.append(_hosted_attempt(model))  # type: ignore[arg-type]
        if _gemini_key():
            attempts.append((GEMINI_BASE_URL, model or _gemini_model(), _gemini_key()))
        if _api_base_url() and _api_key():
            attempts.append((_api_base_url(), model or _api_model(), _api_key()))  # type: ignore[arg-type]
        if _openai_key():
            attempts.append((OPENAI_BASE_URL, model or DEFAULT_OPENAI_MODEL, _openai_key()))

    elif provider == _PROVIDER_GEMINI:
        key = _gemini_key()
        if key:
            attempts.append((GEMINI_BASE_URL, model or _gemini_model(), key))
        # Fallback to generic API then OpenAI if Gemini fails.
        if _api_base_url() and _api_key():
            attempts.append((_api_base_url(), model or _api_model(), _api_key()))  # type: ignore[arg-type]
        if _openai_key():
            attempts.append((OPENAI_BASE_URL, model or DEFAULT_OPENAI_MODEL, _openai_key()))

    elif provider == _PROVIDER_API:
        base = _api_base_url()
        key = _api_key()
        if base and key:
            attempts.append((base, model or _api_model(), key))
        if _openai_key():
            attempts.append((OPENAI_BASE_URL, model or DEFAULT_OPENAI_MODEL, _openai_key()))

    elif provider == _PROVIDER_OPENAI:
        key = _openai_key()
        if key:
            attempts.append((OPENAI_BASE_URL, model or DEFAULT_OPENAI_MODEL, key))

    if not attempts:
        result = _local_fallback(prompt)
        return iter([result]) if stream else result

    last_error: Exception | None = None
    for base_url, req_model, api_key in attempts:
        try:
            logger.debug(
                "Attempting LLM request: base_url=%s model=%s stream=%s",
                base_url,
                req_model,
                stream,
            )
            if stream:
                # Use _stream_completion which keeps the httpx connection open
                # via httpx.stream() context manager while the generator is live.
                return _stream_completion(
                    base_url,
                    req_model,
                    messages,
                    api_key=api_key,
                    timeout=timeout,
                )
            else:
                return _completion_text(
                    base_url,
                    req_model,
                    messages,
                    api_key=api_key,
                    timeout=timeout,
                )

        except (httpx.HTTPError, httpx.TimeoutException, OSError, KeyError, EmptyAnswerError) as exc:
            last_error = exc
            logger.warning(
                "Provider at %s failed: %s. Trying next provider.",
                base_url,
                exc,
            )
            continue

    logger.warning(
        "All LLM providers failed. Last error: %s. Falling back to local mode.",
        last_error,
    )
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
    timeout : float
        HTTP request timeout in seconds.

    Returns
    -------
    str | Iterator[str]
        The LLM response text (or a streaming iterator).
    """
    if provider is None:
        provider = detect_available_provider()

    if provider == _PROVIDER_LOCAL:
        # Extract the last user message for the fallback.
        user_msgs = [m for m in messages if m.get("role") == "user"]
        prompt = user_msgs[-1]["content"] if user_msgs else ""
        result = _local_fallback(prompt)
        return iter([result]) if stream else result

    # Build the ordered list of (base_url, model, api_key) to try.
    attempts: list[tuple[str, str, str | None]] = []

    if provider == _PROVIDER_OLLAMA:
        attempts.append((_ollama_base_url(), model or _ollama_model(), None))
        if _gemini_key():
            attempts.append((GEMINI_BASE_URL, model or _gemini_model(), _gemini_key()))
        if _api_base_url() and _api_key():
            attempts.append((_api_base_url(), model or _api_model(), _api_key()))  # type: ignore[arg-type]
        if _openai_key():
            attempts.append((OPENAI_BASE_URL, model or DEFAULT_OPENAI_MODEL, _openai_key()))
    elif provider == _PROVIDER_GEMINI:
        key = _gemini_key()
        if key:
            attempts.append((GEMINI_BASE_URL, model or _gemini_model(), key))
        if _api_base_url() and _api_key():
            attempts.append((_api_base_url(), model or _api_model(), _api_key()))  # type: ignore[arg-type]
        if _openai_key():
            attempts.append((OPENAI_BASE_URL, model or DEFAULT_OPENAI_MODEL, _openai_key()))
    elif provider == _PROVIDER_API:
        base = _api_base_url()
        key = _api_key()
        if base and key:
            attempts.append((base, model or _api_model(), key))
        if _openai_key():
            attempts.append((OPENAI_BASE_URL, model or DEFAULT_OPENAI_MODEL, _openai_key()))
    elif provider == _PROVIDER_OPENAI:
        key = _openai_key()
        if key:
            attempts.append((OPENAI_BASE_URL, model or DEFAULT_OPENAI_MODEL, key))

    if not attempts:
        user_msgs = [m for m in messages if m.get("role") == "user"]
        prompt = user_msgs[-1]["content"] if user_msgs else ""
        result = _local_fallback(prompt)
        return iter([result]) if stream else result

    for base_url, req_model, api_key in attempts:
        try:
            if stream:
                return _stream_completion(
                    base_url,
                    req_model,
                    messages,
                    api_key=api_key,
                    timeout=timeout,
                )
            else:
                return _completion_text(
                    base_url,
                    req_model,
                    messages,
                    api_key=api_key,
                    timeout=timeout,
                )
        except (httpx.HTTPError, httpx.TimeoutException, OSError, KeyError, EmptyAnswerError) as exc:
            logger.warning("Provider at %s failed: %s", base_url, exc)
            continue

    user_msgs = [m for m in messages if m.get("role") == "user"]
    prompt = user_msgs[-1]["content"] if user_msgs else ""
    result = _local_fallback(prompt)
    return iter([result]) if stream else result


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
