"""Perseus -- the MORIE resident AI agent.

Delegates to the provider-chain in :mod:`morie.llm`.  The LLM module handles
Ollama, Gemini, OpenAI-compatible APIs, direct OpenAI, and a local static
fallback.

When ``stream=True`` is passed to :func:`ask_percy`, the returned dictionary
contains an ``output_stream`` key (an iterator of string chunks) instead of
``output_text``.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

from .cpads import cpads_contract
from .llm import (
    _FallbackText,
    agent_available,  # noqa: F401 -- re-exported
    build_morie_context,
    detect_available_provider,
)
from .llm import ask as llm_ask

logger = logging.getLogger(__name__)


PERSEUS_SYSTEM_PROMPT = """You are Perseus, the MORIE agent for methods for observational inference and robust analysis of interventions in sociolegal studies.
Help users understand datasets, methods, debugging steps, testing strategy, and interpretation.
Be explicit about assumptions, limitations, missing data concerns, and reproducibility risks.
Do not invent data access or approval status for restricted datasets."""


def build_prompt(question: str, context: str | None = None) -> str:
    """Build a prompt from a user question and optional context."""
    prompt = question.strip()
    if context:
        prompt = f"Context:\n{context.strip()}\n\nQuestion:\n{prompt}"
    return prompt


def _how_to_get_a_model() -> str:
    """The three routes to a live model, with the one that applies to this machine first."""
    from .hosted import hosted_failure, hosted_key

    lines = ["To get answers from a model (tried in this order):"]
    if hosted_key() and hosted_failure() == "rejected":
        lines.append(
            "  1. hosted MORIE tier: you are signed in but the gateway rejected the key "
            "(reset, or replaced by a newer sign-in). Run `morie login` again."
        )
    elif hosted_key():
        lines.append(
            "  1. hosted MORIE tier: you are signed in but https://llm.rmorie.com was not "
            "reachable from here. Check the network, then `morie doctor`."
        )
    else:
        lines.append(
            "  1. hosted MORIE tier (free, sign in with GitHub or email): `morie login`; "
            "after that `morie ask` uses it automatically and `morie models` lists what you can ask."
        )
    lines.append(
        "  2. a local Ollama: install it, then `ollama pull gemma4:e2b` (or any model; pick with `morie ask --model NAME`)."
    )
    lines.append(
        "  3. your own model: `morie provider set --base-url URL --key KEY [--model NAME]` for any "
        "OpenAI-compatible endpoint (OpenAI, Anthropic's https://api.anthropic.com/v1, OpenRouter, "
        "LM Studio, vLLM ...), or GEMINI_API_KEY / OPENAI_API_KEY in the environment."
    )
    lines.append(
        "`morie doctor` shows which of these answers from this machine. Until one does, "
        "`morie list-modules`, `morie explain FILE` and `morie cheatsheet` describe the modules, "
        "their outputs and the commands without a model."
    )
    return "\n".join(lines)


def _local_fallback_response(question: str, context: str | None = None) -> str:
    """Generate a local keyword-matched response when no LLM is available."""
    normalized = question.lower()
    sections = [
        "MORIE local agent mode is active because no live LLM provider was detected.",
    ]

    if "ipw" in normalized or "propensity" in normalized:
        sections.append(
            "MORIE includes propensity-score and eBAC IPW workflows in both the package interfaces "
            "and the module runners. Use the Python and R module guides to find the relevant "
            "entrypoints and expected outputs."
        )
    if "cpads" in normalized or "dataset" in normalized or "data" in normalized:
        contract = cpads_contract()
        sections.append(
            "CPADS is treated as a local private dataset. Required canonical variables: "
            + ", ".join(contract["required_variables"])
            + ". Expected wrangled path pattern: "
            + str(contract["expected_wrangled_path"])
        )

    if context:
        sections.append(f"Context noted: {context.strip()}")

    sections.append(_how_to_get_a_model())
    return "\n\n".join(sections)


def _try_agent(question: str, *, stream: bool = False, model: str | None = None) -> dict[str, Any] | None:
    """Try the agentic path (Ollama native tool calling). Returns None if unavailable."""
    try:
        from .agent import create_agent
    except ImportError:
        return None

    try:
        agent = create_agent(model=model)
        if stream:
            return {
                "mode": "agent",
                "model": agent._model,
                "output_stream": agent.chat_stream(question),
            }
        resp = agent.chat(question)
        suffix = ""
        if resp.tool_calls_made:
            suffix = f"\n\n[{len(resp.tool_calls_made)} tool calls in {resp.iterations} iterations]"
        return {
            "mode": "agent",
            "model": resp.model,
            "output_text": resp.text + suffix,
            "tool_calls": resp.tool_calls_made,
            "failed": bool(getattr(resp, "failed", False)),
        }
    except Exception as exc:
        logger.debug("Agent path failed: %s", exc)
        return None


def ask_percy(
    question: str,
    *,
    context: str | None = None,
    model: str | None = None,
    system_prompt: str = PERSEUS_SYSTEM_PROMPT,
    allow_fallback: bool = True,
    stream: bool = False,
    use_agent: bool = True,
) -> dict[str, Any]:
    """Query Perseus via the LLM provider chain.

    When ``use_agent=True`` (default) and Ollama is available, Perseus uses the
    full agentic loop with 13 tools (search, execute, read/write, shell, data).
    Falls back to simple LLM chat or static text when tools are unavailable.

    Returns a dict with ``mode``, ``model``, and either ``output_text`` (str)
    or ``output_stream`` (Iterator[str]).
    """
    provider = detect_available_provider()

    if use_agent and provider == "ollama":
        agent_result = _try_agent(question, stream=stream, model=model)
        if agent_result is not None:
            return agent_result

    if provider == "local":
        if allow_fallback:
            fallback_text = _local_fallback_response(question, context=context)
            result: dict[str, Any] = {
                "mode": "local_fallback",
                "model": "local",
            }
            if stream:
                result["output_stream"] = iter([fallback_text])
            else:
                result["output_text"] = fallback_text
            return result
        raise RuntimeError("No LLM provider is available. Install Ollama, set LLM_API_KEY, or set OPENAI_API_KEY.")

    morie_context = build_morie_context()

    try:
        full_prompt = build_prompt(question, context=context)
        output = llm_ask(
            full_prompt,
            context=morie_context,
            stream=stream,
            model=model,
            provider=provider,
            system_prompt=system_prompt,
        )

        # llm.ask() has its own static fallback for a provider that fails
        # mid-request; it returns the same text a local-only session gets.
        # Report it as such so the exit code says nobody answered.
        if stream:
            first = next(output, None)
            fallback = isinstance(first, _FallbackText)
            output = itertools.chain([] if first is None else [first], output)
        else:
            assert isinstance(output, str)
            fallback = isinstance(output, _FallbackText)
        result = {
            "mode": "local_fallback" if fallback else "live_api",
            "model": "local" if fallback else (model or provider),
        }
        if stream:
            result["output_stream"] = output
        else:
            result["output_text"] = output
        return result

    except Exception as exc:
        logger.warning("LLM request failed: %s", exc)
        if allow_fallback:
            fallback_text = _local_fallback_response(question, context=context)
            result = {
                "mode": "local_fallback",
                "model": "local",
            }
            if stream:
                result["output_stream"] = iter([fallback_text])
            else:
                result["output_text"] = fallback_text
            return result
        raise


# Backward compatibility aliases
ask_morie_assistant = ask_percy
morie_ask_percy = ask_percy  # R-package bridge (perseus.R imports this name)
assistant_available = agent_available
build_assistant_prompt = build_prompt
DEFAULT_SYSTEM_PROMPT = PERSEUS_SYSTEM_PROMPT
