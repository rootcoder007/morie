# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.14: the Co-Occurrence Bias Score."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch6_co_occurrence_bias"]


def _conditional(w, A, name):
    if isinstance(A, dict):
        counts = {k: float(v) for k, v in A.items()}
    else:
        toks = []
        for item in A:
            toks.extend(item.split() if isinstance(item, str) else [item])
        counts = {}
        for t in toks:
            counts[t] = counts.get(t, 0.0) + 1.0
    total = float(sum(counts.values()))
    if total <= 0:
        raise ValueError(f"{name} contains no tokens.")
    c = counts.get(w, 0.0)
    if c <= 0:
        raise ValueError(
            f"{w!r} never co-occurs with {name}; log 0 is undefined, so "
            "the score does not exist for this token.")
    return c / total, c, total


def kamath_ch6_co_occurrence_bias(w, A_i, A_j):
    """score(w) = log[P(w | A_i) / P(w | A_j)].

    P(w | A) is w's share of the tokens generated alongside attribute
    set A. A token absent from either side has an undefined score --
    raised, not silently returned as -inf, because "never seen" and
    "seen rarely" are different findings. 0 means the token is equally
    associated with both groups.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.14, printed
    p. 236.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch6_co_occurrence_bias(
    ...     "nurse", ["nurse nurse doctor"], ["nurse doctor doctor"])
    >>> abs(out["estimate"] - math.log(2.0)) < 1e-12
    True
    """
    pi, ci, ti = _conditional(w, A_i, "A_i")
    pj, cj, tj = _conditional(w, A_j, "A_j")
    return RichResult(payload={
        "estimate": float(np.log(pi / pj)), "p_given_Ai": pi,
        "p_given_Aj": pj, "count_Ai": ci, "count_Aj": cj,
        "n": int(ti + tj),
        "method": "Co-Occurrence Bias Score (Kamath Eq 6.14)"})


def cheatsheet():
    return "km090: log P(w|A_i) / P(w|A_j) over generated text"
