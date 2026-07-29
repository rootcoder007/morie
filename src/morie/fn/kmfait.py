# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""RAGAS faithfulness: fraction of answer claims entailed by the
retrieved context."""

import re

from ._richresult import RichResult

__all__ = ["kamath_ragas_faithfulness"]

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text):
    return _WORD.findall(str(text).lower())


def _split_claims(answer):
    """A string is split into sentences; a sequence is taken as-is."""
    if isinstance(answer, str):
        parts = [p.strip() for p in re.split(r"[.!?\n]+", answer)]
        return [p for p in parts if p]
    return [str(a).strip() for a in answer if str(a).strip()]


def _lexical_entailment(claim, context_tokens):
    """Default judge: every content token of the claim occurs in the
    context. Deliberately strict -- it under-reports rather than
    hallucinating support, and callers with a real NLI model pass
    ``entails``."""
    toks = _tokens(claim)
    if not toks:
        raise ValueError("a claim contains no tokens at all.")
    return all(t in context_tokens for t in toks)


def kamath_ragas_faithfulness(answer, context, entails=None):
    """faithfulness = |supported claims| / |total claims|.

    ``answer`` is a sequence of claims, or a string that is split into
    sentences. ``context`` is the retrieved passage (string or
    sequence of strings). ``entails`` is an optional judge
    ``(claim, context) -> bool``; without one a strict lexical
    containment test is used and that choice is reported in the
    payload, because "faithfulness 0.9" means nothing without knowing
    who decided.

    Reference: the worklist cites Kamath, Keenan, Somers and Sorenson
    (2024), *Large Language Models: A Deep Dive*, Springer, Ch 7, RAG
    faithfulness. Implemented from the ratio given in the spec line.

    Examples
    --------
    >>> out = kamath_ragas_faithfulness(
    ...     ["the sky is blue", "the sky is green"], "the sky is blue today")
    >>> out["estimate"]
    0.5
    >>> out["supported"]
    1
    """
    claims = _split_claims(answer)
    if not claims:
        raise ValueError(
            "the answer contains no claims; faithfulness is 0/0 and "
            "calling that 1.0 would be a decision, not a measurement.")
    if isinstance(context, str):
        ctx_text = context
    else:
        ctx_text = " ".join(str(c) for c in context)
    if not _tokens(ctx_text):
        raise ValueError(
            "the context is empty; every claim would be unsupported by "
            "construction.")
    if entails is not None and not callable(entails):
        raise ValueError("entails must be callable (claim, context) -> bool.")

    ctx_tokens = set(_tokens(ctx_text))
    flags = []
    for c in claims:
        if entails is None:
            ok = _lexical_entailment(c, ctx_tokens)
        else:
            ok = entails(c, ctx_text)
            if not isinstance(ok, (bool,)) and ok not in (0, 1):
                raise ValueError(
                    "entails must return a bool; got "
                    f"{type(ok).__name__}.")
        flags.append(bool(ok))
    supported = int(sum(flags))
    return RichResult(payload={
        "estimate": supported / len(claims),
        "supported": supported,
        "n_claims": len(claims),
        "claim_supported": flags,
        "claims": claims,
        "judge": "caller-supplied" if entails else "lexical containment",
        "n": len(claims),
        "method": "RAGAS faithfulness (supported claims / total claims)"})


def cheatsheet():
    return "kmfait: supported answer claims / total claims, judge reported"
