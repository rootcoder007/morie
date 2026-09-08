# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 7: Corrective RAG's retrieval-confidence router."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_corrective_rag"]


def kamath_corrective_rag(query, docs, clf, tau_hi, tau_lo):
    r"""Route on s = clf(q, d): correct, incorrect, or ambiguous.

    Every retrieved document is scored; the decision uses the BEST
    score, as CRAG's "at least one relevant document" rule requires:
    ``>= tau_hi`` keeps the retrieved context ("correct"), ``<= tau_lo``
    discards it for a web search ("incorrect"), anything between takes
    both ("ambiguous"). The kept context is the documents that
    themselves clear ``tau_hi``.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 7, Corrective RAG;
    Yan et al. (2024).

    Examples
    --------
    >>> f = lambda q, d: 0.9 if d == "d1" else 0.1
    >>> out = kamath_corrective_rag("q", ["d1", "d2"], f, 0.8, 0.2)
    >>> out["action"], out["ctx"]
    ('use_docs', ['d1'])
    """
    if not callable(clf):
        raise ValueError("clf must be callable clf(query, doc) -> "
                         "confidence.")
    D = list(docs)
    if len(D) == 0:
        raise ValueError("no documents were retrieved; there is "
                         "nothing to grade.")
    hi = float(tau_hi)
    lo = float(tau_lo)
    if lo > hi:
        raise ValueError(
            f"tau_lo = {lo} exceeds tau_hi = {hi}; the ambiguous band "
            "would be empty and inverted.")
    s = np.array([float(clf(query, d)) for d in D])
    if not np.all(np.isfinite(s)):
        raise ValueError("clf returned a non-finite confidence.")
    best = float(s.max())
    if best >= hi:
        action = "use_docs"
    elif best <= lo:
        action = "fallback_web"
    else:
        action = "mixed"
    keep = [d for d, v in zip(D, s) if v >= hi]
    ctx = keep if action == "use_docs" else (
        [] if action == "fallback_web" else
        [d for d, v in zip(D, s) if v > lo])
    return RichResult(payload={
        "estimate": best, "action": action, "ctx": ctx,
        "scores": [float(v) for v in s], "best_score": best,
        "tau_hi": hi, "tau_lo": lo, "n": len(D),
        "method": "Corrective RAG retrieval router (Kamath Ch 7)"})


def cheatsheet():
    return "kmcrag: grade retrieved docs, then keep / web-search / both"
