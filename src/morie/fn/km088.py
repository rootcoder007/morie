# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.12: the Context Association Test score."""

import numpy as np

from ._richresult import RichResult
from .km086 import _log_probs

__all__ = ["kamath_ch6_cat_metric"]


def kamath_ch6_cat_metric(M, U, theta=None):
    """CAT(S) = (1/|M|) sum_{m in M} log P(m | U ; theta).

    The MIRROR of CrowS-Pairs: CAT scores the modified tokens M given
    the unmodified context U -- P(M|U) rather than P(U|M) -- and it
    AVERAGES where Eq 6.11 sums, so the two are not comparable
    sentence to sentence. ``theta`` is a callable (M, U, i) ->
    probability; with None, ``M`` holds those probabilities already.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.12, printed
    p. 236.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch6_cat_metric([0.5, 0.25], ["the", "doctor"])
    >>> abs(out["estimate"] + math.log(8.0) / 2) < 1e-12
    True
    >>> kamath_ch6_cat_metric([1.0], ["ctx"])["estimate"]
    0.0
    """
    ctx = list(U)
    if not ctx:
        raise ValueError("U is empty; CAT conditions on the unmodified "
                         "context, so there must be at least one token.")
    if theta is not None and not callable(theta):
        raise ValueError("theta must be a callable (M, U, i) -> "
                         "probability, or None.")
    toks = list(M)
    scorer = None if theta is None else (lambda i: theta(toks, ctx, i))
    logs, seq = _log_probs(toks, scorer, "M")
    return RichResult(payload={
        "estimate": float(logs.mean()),
        "per_token": [float(v) for v in logs],
        "n_context": len(ctx), "n": len(seq),
        "method": "Context Association Test score (Kamath Eq 6.12)"})


def cheatsheet():
    return "km088: mean log P(modified token | unmodified context)"
