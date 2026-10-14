"""Language models for MORIE: the one you bring.

Nothing is hosted by the project and no prompt goes anywhere you did not set up. ``ask`` uses,
in the automatic order (``route = auto``), the same one rmoriebricklayer uses:

    1. own       -- your own OpenAI-compatible server (LM Studio, vLLM, llama.cpp, a gateway)
    2. provider  -- the first provider whose key is set: OpenAI, Anthropic, Google Gemini,
                    OpenRouter, Groq, Together, Fireworks, DeepSeek, Mistral, xAI, Cerebras,
                    Perplexity, Hugging Face, OpenCode Zen, Cohere, NVIDIA (``morie.providers``)
    3. ollama    -- an Ollama server (``ollama serve``) with a model pulled
    4. local     -- MORIE's keyword help, marked as such, no network

Anthropic's Messages API and Gemini's ``generateContent`` are spoken natively; the rest are
OpenAI-compatible (``httpx``).

Environment Variables
---------------------
ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, ... : str
    A provider's key (``morie.providers.PRESETS`` lists every one); its model is
    ``MORIE_<PROVIDER>_MODEL``, else the provider's default.
MORIE_LLM_ROUTE : str
    ``auto`` (default), ``own``, ``ollama`` or a provider id: force the route ``ask`` takes.
OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_API_KEY : str
    The Ollama server (``OLLAMA_BASE_URL`` is still read), its model and an optional key
    (``OLLAMA_HOST=https://ollama.com`` with an Ollama key reaches Ollama's own cloud).
MORIE_LLM_BASE_URL, MORIE_LLM_API_KEY, MORIE_LLM_MODEL : str
    Your own OpenAI-compatible server (``LLM_API_BASE_URL`` / ``LLM_API_KEY`` are still read).

Every one of these can be saved instead with ``morie config set KEY VALUE`` or
:func:`config` (``$XDG_CONFIG_HOME/morie/llm.json``, the file rmoriebricklayer's ``rmbl config``
writes too); a variable that is set wins over the saved value. ``morie config help`` lists them,
``morie doctor`` says which route ``ask`` takes.

References
----------
* Ollama API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
* Anthropic Messages API: https://docs.anthropic.com/en/api/messages
* Gemini generateContent: https://ai.google.dev/api/generate-content
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
from . import providers as _providers
from .cpads import cpads_contract
from .llm_config import config, config_get, config_path, config_unset  # noqa: F401 -- public API
from .modules import MODULE_SPECS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = ""  # Auto-detected from running Ollama instance
DEFAULT_API_MODEL = "google/gemma-3-27b-it"

_PROVIDER_OLLAMA = "ollama"
_PROVIDER_API = "own"  # your own OpenAI-compatible endpoint ("api" is still accepted)
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


_ollama_cached: bool | None = None


def _probe_ollama(timeout: float = _PROBE_TIMEOUT) -> bool:
    """Return True if a local Ollama instance answers AND has a model to use.

    A server with nothing pulled cannot answer a question, so it is not a usable
    route unless a model is named (``OLLAMA_MODEL`` or ``morie config set
    ollama.model``): the automatic order then ends with MORIE's keyword help instead
    of stopping at an Ollama that can only fail.

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
    if r == "api":
        r = "own"
    if r == "hosted":
        raise ValueError(_providers.RETIRED)
    if r not in _cfg.ROUTES:
        raise ValueError(f"route must be one of: {', '.join(_cfg.ROUTES)}")
    return r


def detect_available_provider(route: str | None = None) -> str:
    """Which language model ``ask`` would use: the one you brought.

    ``route`` (else ``MORIE_LLM_ROUTE``, else the route saved with ``morie config set route``)
    forces one: ``"own"`` (your OpenAI-compatible endpoint), ``"ollama"``, or a provider id
    (``"anthropic"``, ``"openai"``, ``"gemini"``, ``"groq"``, ...; ``morie.providers.IDS``). With
    ``"auto"`` (the default) the order is the one rmoriebricklayer uses:

    1. **own**     -- your own endpoint is set (``morie config set own.url``, ``MORIE_LLM_BASE_URL``).
    2. a **provider** whose key is set (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ... or
       ``morie config set <provider>.key``), in the order of ``morie.providers.PRESETS``.
    3. **ollama**  -- an Ollama server answers and has a model (one pulled, or ``OLLAMA_MODEL``).
    4. **local**   -- nothing is set up: the answer is MORIE's keyword help, marked as such.

    Nothing is hosted by the project: no prompt goes anywhere you did not set up.

    Returns
    -------
    str
        ``"own"``, ``"ollama"``, a provider id, or ``"local"``.

    Examples
    --------
    >>> detect_available_provider() in ("own", "ollama", "local", *__import__("morie.providers").providers.IDS)
    True
    """
    r = _route_name(route)
    if r == "ollama":
        # forced: used even with nothing pulled (the request then says what is missing)
        return _PROVIDER_OLLAMA if _ollama_base_url() else _PROVIDER_LOCAL
    if r == "own":
        return _PROVIDER_API if _own_ready() else _PROVIDER_LOCAL
    if r in _providers.IDS:
        return r if _providers.key(r) else _PROVIDER_LOCAL
    if _own_ready():
        return _PROVIDER_API
    keyed = _providers.with_key()
    if keyed:
        return keyed[0]
    if _ollama_base_url() and _probe_ollama():
        return _PROVIDER_OLLAMA
    return _PROVIDER_LOCAL


def _provider_model(provider: str) -> str:
    """The model a provider would be asked with (no network beyond the probes already made)."""
    if provider == _PROVIDER_OLLAMA:
        return _ollama_model()
    if provider == _PROVIDER_API:
        return _api_model()
    if provider in _providers.IDS:
        return _providers.model(provider)
    return ""


def _provider_label(provider: str) -> str:
    if provider == _PROVIDER_OLLAMA:
        return "Ollama"
    if provider == _PROVIDER_API:
        return "own endpoint"
    if provider in _providers.IDS:
        return _providers.preset(provider).label
    return provider


def detect_provider_and_model(route: str | None = None) -> tuple[str, str]:
    """Detect LLM provider and return (provider, human-readable model label).

    Returns
    -------
    tuple[str, str]
        ``(provider_key, display_label)`` -- e.g. ``("anthropic", "Anthropic:claude-opus-5-5")``.
    """
    provider = detect_available_provider(route)
    if provider == _PROVIDER_LOCAL:
        return provider, "local fallback (no LLM)"
    return provider, f"{_provider_label(provider)}:{_provider_model(provider)}"


def detect_model_display() -> dict[str, str]:
    """Return display info with inner (family:size) and outer (model name).

    Returns
    -------
    dict[str, str]
        Keys: ``inner``, ``outer``, ``model``, ``provider``.
    """
    provider, label = detect_provider_and_model()
    model = label.split(":", 1)[1] if ":" in label else label
    return {"inner": model, "outer": model, "model": model, "provider": provider}


def route_status() -> list[dict[str, str]]:
    """One row per route (own endpoint, each provider with a key, local Ollama): ``route``, ``status``, ``detail``."""
    rows = []
    own = _api_base_url()
    rows.append(
        {"route": "own endpoint", "status": "configured", "detail": f"{own}  model: {_api_model()}"}
        if own
        else {
            "route": "own endpoint",
            "status": "not set",
            "detail": "`morie config set own.url URL` (and own.model, own.key) for any OpenAI-compatible server",
        }
    )
    keyed = _providers.with_key()
    for pid in keyed:
        p = _providers.preset(pid)
        rows.append({"route": p.label, "status": "key set", "detail": f"{p.base}  model: {_providers.model(pid)}"})
    rest = [p for p in _providers.PRESETS if p.id not in keyed]
    if rest:
        rows.append(
            {
                "route": "other providers",
                "status": "no key",
                "detail": ", ".join(f"{p.id} ({p.key_env})" for p in rest),
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
                    "detail": f"{base} (start it with `ollama serve`; or install Ollama and pull a model, "
                    "or `morie config set ollama.url ADDRESS`)",
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
    return rows


def route_summary(route: str | None = None) -> str:
    """One line on the route ``ask`` takes now, e.g. ``ask uses: Anthropic, model claude-opus-5-5``."""
    r = _route_name(route)
    provider = detect_available_provider(r)
    forced = "" if r == "auto" else f"  (route = {r})"
    if provider == _PROVIDER_LOCAL:
        if r == "auto":
            return (
                "ask has no route yet: set a provider's key (e.g. ANTHROPIC_API_KEY), start Ollama "
                "(`ollama serve`) with a model, or `morie config setup`"
            )
        return f"ask has no route: route = {r} is not set up here (`morie config` shows the settings)"
    model = _provider_model(provider) or "(none)"
    return f"ask uses: {_provider_label(provider)}, model {model}{forced}"


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
    if base in {p.base for p in _providers.PRESETS}:
        return f"{base}/chat/completions"  # a provider's documented base, as it is (Perplexity has no /v1)
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

    # reasoning models spend tokens thinking before answering: give them room,
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

    # reasoning models spend tokens thinking before answering: give them room,
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

To get answers from a model, bring one (tried in this order):

  1. Your own OpenAI-compatible server (LM Studio, vLLM, llama.cpp, a gateway):
       morie config set own.url http://localhost:1234/v1

  2. A provider's key (OpenAI, Anthropic, Gemini, OpenRouter, Groq, Mistral and more):
       export ANTHROPIC_API_KEY="..."       # or: morie config set anthropic.key
       export OPENAI_API_KEY="..."          # any provider: morie config help lists them

  3. A local Ollama (private, no key):
       curl -fsSL https://ollama.com/install.sh | sh
       ollama serve                         # if it is not running as a service
       ollama pull gemma4:e2b

`morie config setup` walks through it; `morie doctor` reports what is reachable from here.
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


def _tagged_attempts(provider: str, model: str | None) -> list[tuple[str, str, str, str, str | None]]:
    """The ordered (route, api, base_url, model, api_key) attempts, starting at ``provider``.

    ``route`` is ``own``, ``ollama`` or a provider id; ``api`` is ``openai``, ``anthropic`` or
    ``gemini``. After the first choice come the other routes that are set up, in the automatic
    order, so a refused key or a dead server falls through to the next. An attempt with no model
    to ask is left out: it could only fail.
    """
    order: list[str] = []
    if _own_ready():
        order.append(_PROVIDER_API)
    order += _providers.with_key()
    if _ollama_base_url() and (provider == _PROVIDER_OLLAMA or _probe_ollama()):
        order.append(_PROVIDER_OLLAMA)
    if provider in order:
        order.remove(provider)
        order.insert(0, provider)
    elif provider == _PROVIDER_OLLAMA and _ollama_base_url():
        order.insert(0, provider)
    attempts: list[tuple[str, str, str, str, str | None]] = []
    first = True
    for r in order:
        m = model if first else None  # a named model belongs to the route it was named for
        first = False
        if r == _PROVIDER_API:
            attempts.append((r, "openai", _api_base_url(), m or _api_model(), _api_key()))  # type: ignore[arg-type]
        elif r == _PROVIDER_OLLAMA:
            attempts.append((r, "openai", _ollama_base_url(), m or _ollama_model(), _ollama_key()))  # type: ignore[arg-type]
        else:
            pr = _providers.preset(r)
            attempts.append((r, pr.api, pr.base, m or _providers.model(r), _providers.key(r)))
    return [a for a in attempts if a[3] and a[2]]


def _provider_attempts(provider: str, model: str | None) -> list[tuple[str, str, str, str | None]]:
    """The ordered (api, base_url, model, api_key) attempts, the same for ask() and ask_multi()."""
    return [a[1:] for a in _tagged_attempts(provider, model)]


class NoModelError(ValueError):
    """A provider whose key is set but whose model is not: said with how to name one."""


def _attempts_for(provider: str, model: str | None, route: str) -> list[tuple[str, str, str, str | None]]:
    """The attempts, kept to the forced route when one is set (no falling back to another route).

    A provider with a key and no model (one without a default, and no ``MORIE_<ID>_MODEL`` or
    ``<id>.model``) is an error that says how to name one, as rmoriebricklayer says it: asking it
    could only fail, and falling through would hide why."""
    if provider in _providers.IDS and not (model or _providers.model(provider)):
        raise NoModelError(
            f"{_providers.preset(provider).label} has no model set: `morie config set {provider}.model NAME` "
            f"(or {_providers.model_env(provider)}); `morie models --route {provider}` lists them"
        )
    tagged = _tagged_attempts(provider, model)
    if route != "auto":
        tagged = [a for a in tagged if a[0] == route]
    return [a[1:] for a in tagged]


def _run_attempts(
    attempts: list[tuple[str, str, str, str | None]],
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
    for api, base_url, req_model, api_key in attempts:
        try:
            logger.debug("Attempting LLM request: base_url=%s model=%s stream=%s", base_url, req_model, stream)
            if api != "openai":
                # Anthropic's and Gemini's own APIs: one answer, handed over whole when streaming
                text = _providers.chat_native(api, base_url, api_key, req_model, messages, timeout=timeout)
                if not text.strip():
                    raise EmptyAnswerError(f"{req_model} at {base_url} returned no text")
                return iter([text]) if stream else text
            if stream:
                # _stream_completion keeps the httpx connection open while the generator is live
                gen = iter(_stream_completion(base_url, req_model, messages, api_key=api_key, timeout=timeout))
                first = next(gen, None)
                if first is None:
                    raise EmptyAnswerError(f"{req_model} at {base_url} streamed no text")
                return itertools.chain([first], gen)
            return _completion_text(base_url, req_model, messages, api_key=api_key, timeout=timeout)
        except (httpx.HTTPError, httpx.TimeoutException, OSError, KeyError, EmptyAnswerError) as exc:
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

    The route chain is the one you brought: your own OpenAI-compatible endpoint -> each
    provider whose key is set -> Ollama -> MORIE's keyword help. Each is tried in order; on
    failure the next is attempted (a forced ``route`` never falls back to another one).

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
        Force a specific provider (``"own"``, ``"ollama"``, a provider id such as
        ``"anthropic"``, or ``"local"``).  When ``None``, :func:`detect_available_provider`
        is used to auto-detect.
    route : str | None
        ``"auto"``, ``"own"``, ``"ollama"`` or a provider id. When ``None``, the
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
    Needs a model you brought (a provider's key, your own server, or a local Ollama):

    >>> # Non-streaming (returns full text)
    >>> response = ask("What is AIPW?")  # doctest: +SKIP
    >>> isinstance(response, str)  # doctest: +SKIP
    True

    >>> # Streaming: chunks print as they arrive
    >>> for chunk in ask("Explain TMLE", stream=True):  # doctest: +SKIP
    ...     print(chunk, end="")
    """
    r = _route_name(route)
    if provider == "api":
        provider = _PROVIDER_API
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
        ``"auto"``, ``"own"``, ``"ollama"`` or a provider id; ``None`` uses the saved route.
    timeout : float
        HTTP request timeout in seconds.

    Returns
    -------
    str | Iterator[str]
        The LLM response text (or a streaming iterator).
    """
    r = _route_name(route)
    if provider == "api":
        provider = _PROVIDER_API
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
