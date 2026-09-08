# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.1: FActScore, atomic-fact precision under abstention."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch6_factscore"]


def kamath_ch6_factscore(M, X, A_y, C):
    """FActScore(M) = E_x[(1/|A_y|) sum_a I[a supported by C] | M_x responds].

    The expectation is CONDITIONED on the model responding, so prompts
    the model declines (``M`` returning None) are dropped from the
    average rather than scored 0 -- that conditioning is the whole
    point of the metric and is reported as ``response_rate``.

    ``M`` is a callable prompt -> response or None; ``A_y`` a callable
    response -> list of atomic facts; ``C`` the knowledge source, a
    callable fact -> bool or any container supporting ``in``.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.1, printed
    p. 221.

    Examples
    --------
    >>> M = lambda x: x or None
    >>> out = kamath_ch6_factscore(M, ["a b", "c", ""], str.split,
    ...                            {"a", "c"})
    >>> out["estimate"], out["n_responded"]
    (0.75, 2)
    >>> round(out["response_rate"], 10)
    0.6666666667
    """
    prompts = list(X)
    if not prompts:
        raise ValueError("X is empty; an expectation over no prompts is "
                         "undefined.")
    if not callable(M):
        raise ValueError("M must be a callable prompt -> response or None.")
    if not callable(A_y):
        raise ValueError("A_y must be a callable response -> atomic facts.")
    supported = C if callable(C) else (lambda a: a in C)
    per_prompt = []
    for x in prompts:
        resp = M(x)
        if resp is None:
            continue
        facts = list(A_y(resp))
        if not facts:
            raise ValueError(
                f"the response to {x!r} yielded no atomic facts; 1/|A_y| "
                "is undefined.")
        flags = [1.0 if supported(a) else 0.0 for a in facts]
        per_prompt.append(float(np.mean(flags)))
    if not per_prompt:
        raise ValueError("the model responded to no prompt; the "
                         "conditional expectation is undefined.")
    arr = np.asarray(per_prompt, dtype=float)
    return RichResult(payload={
        "estimate": float(arr.mean()), "per_prompt": per_prompt,
        "n_responded": len(per_prompt),
        "response_rate": len(per_prompt) / len(prompts),
        "n": len(prompts),
        "method": "FActScore (Kamath Eq 6.1)"})


def cheatsheet():
    return "km077: mean supported-fact fraction over answered prompts"
