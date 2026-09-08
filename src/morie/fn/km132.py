# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 9.4: LLM text output plus modality signal tokens."""

from ._richresult import RichResult

__all__ = ["kamath_ch9_llm_signal_tokens"]


def kamath_ch9_llm_signal_tokens(P_X, F_T, llm=None):
    r"""(t, S_X) = LLM(P_X, F_T).

    The LLM in an MMLLM returns BOTH text ``t`` and the signal tokens
    ``S_X`` that instruct the generator; the contract enforced here is
    exactly that -- ``llm`` must return a 2-tuple, and the signal
    tokens must be a sequence. ``estimate`` is the number of signal
    tokens emitted, which is what downstream routing branches on.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, Eq 9.4, printed
    p. 383.

    Examples
    --------
    >>> out = kamath_ch9_llm_signal_tokens(
    ...     [[0.0]], [[1.0]], llm=lambda p, f: ("a cat", ["<IMG>"]))
    >>> out["text"], out["signal_tokens"], out["estimate"]
    ('a cat', ['<IMG>'], 1)
    """
    if not callable(llm):
        raise ValueError("llm= must be a callable LLM(P_X, F_T).")
    got = llm(P_X, F_T)
    if not (isinstance(got, tuple) and len(got) == 2):
        raise ValueError("the LLM must return the 2-tuple (t, S_X) of "
                         "Eq 9.4; got "
                         f"{type(got).__name__}.")
    t, S_X = got
    try:
        signals = list(S_X)
    except TypeError:
        raise ValueError("S_X must be a sequence of signal tokens.") \
            from None
    return RichResult(payload={
        "estimate": len(signals), "text": t, "signal_tokens": signals,
        "generates_modality": bool(signals), "n": len(signals),
        "method": "LLM text and signal tokens (Kamath Eq 9.4)"})


def cheatsheet():
    return "km132: (t, S_X) = LLM(P_X, F_T), signal-token count reported"
