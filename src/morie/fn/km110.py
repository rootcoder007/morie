# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 7.1: reciprocal rank fusion (RRF) score."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch7_rrf_score"]


def kamath_ch7_rrf_score(r, k=60):
    r"""Score_RRF = 1 / (r + 60) for a document at rank ``r``.

    ``r`` is a document's 1-based place in one or more search
    rankings; RAG-Fusion merges rankings by SUMMING the per-ranking
    scores, so ``estimate`` is that sum and ``scores`` the parts.
    The constant 60 is Cormack et al.'s damping term, exposed as
    ``k`` but defaulted to the book's value.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 7, Eq 7.1, printed
    p. 285.

    Examples
    --------
    >>> out = kamath_ch7_rrf_score([1, 2])
    >>> round(out["estimate"], 6)          # 1/61 + 1/62
    0.032522
    >>> round(out["scores"][0], 6)
    0.016393
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    if r.size == 0:
        raise ValueError("no ranks given; RRF over an empty ranking "
                         "list is undefined.")
    if np.any(r < 1):
        raise ValueError("ranks are 1-based; a rank below 1 is not a "
                         "search position.")
    if np.any(np.abs(r + k) < 1e-12):
        raise ValueError(f"r + k = 0 for some rank with k = {k}; the "
                         "RRF score is a pole there.")
    scores = 1.0 / (r + float(k))
    return RichResult(payload={
        "estimate": float(scores.sum()),
        "scores": [float(v) for v in scores],
        "ranks": [float(v) for v in r], "k": float(k), "n": int(r.size),
        "method": "reciprocal rank fusion score (Kamath Eq 7.1)"})


def cheatsheet():
    return "km110: 1/(r+60) per ranking, summed across search variants"
