# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Groundedness reward: fraction of answer tokens supported by the
retrieved context."""

from ._richresult import RichResult

__all__ = ["kamath_groundedness_reward"]


def kamath_groundedness_reward(y_tokens, ctx_tokens, lowercase=True):
    """grd(y | ctx) = |{tokens in y also in ctx}| / |y|.

    Every occurrence in ``y`` is counted (the denominator is the answer
    LENGTH, not its vocabulary size), and membership is tested against
    the context's token SET, exactly as the spec line reads.

    Reference: the worklist cites Kamath, Keenan, Somers and Sorenson
    (2024), *Large Language Models: A Deep Dive*, Springer, Ch 5,
    groundedness reward modelling.

    Examples
    --------
    >>> out = kamath_groundedness_reward(["a", "b", "c"], ["a", "c"])
    >>> abs(out["estimate"] - 2 / 3) < 1e-12
    True
    >>> out["n_grounded"]
    2
    >>> kamath_groundedness_reward(["a", "a"], ["a"])["estimate"]
    1.0
    """
    y = list(y_tokens)
    ctx = list(ctx_tokens)
    if not y:
        raise ValueError(
            "the answer has no tokens; groundedness is 0/0 and a reward "
            "for saying nothing is undefined, not 1.")
    if not ctx:
        raise ValueError(
            "the context has no tokens; every answer token would be "
            "ungrounded by construction.")
    if lowercase:
        y = [str(t).lower() for t in y]
        ctx = [str(t).lower() for t in ctx]
    else:
        y = [str(t) for t in y]
        ctx = [str(t) for t in ctx]
    cset = set(ctx)
    flags = [t in cset for t in y]
    hit = int(sum(flags))
    return RichResult(payload={
        "estimate": hit / len(y),
        "n_grounded": hit,
        "n_tokens": len(y),
        "ungrounded": sorted({t for t, ok in zip(y, flags) if not ok}),
        "n": len(y),
        "method": "Groundedness reward (answer tokens found in context)"})


def cheatsheet():
    return "kmgrnd: |answer tokens present in ctx| / |answer tokens|"
