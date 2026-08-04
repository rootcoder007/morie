# morie.fn -- function file (rootcoder007/morie)
"""Mean reciprocal rank."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["mrr"]


def mrr(pred_rank, relevant):
    """Rank-based score that only cares where the first hit lands.

    The reciprocal makes the metric brutally top-heavy: moving the first
    relevant item from rank two to rank one is worth as much as moving
    it from rank ten to rank two.  That is deliberate, and it is why MRR
    suits question answering, where there is one right answer, and suits
    recommendation badly, where there are many.

    Formula: ``MRR = (1 / |Q|) sum_q 1 / rank_q``, with a query
    contributing zero when it has no relevant item ranked.

    Parameters
    ----------
    pred_rank : array-like, shape (Q, k)
        Ranked item ids per query, best first.
    relevant : array-like, shape (Q,) or (Q, r)
        Relevant item id(s) per query.

    Returns
    -------
    RichResult
        ``estimate`` (MRR), ``rr`` (per query), ``n_hit``, ``Q``.

    References
    ----------
    Voorhees, E. M. (1999).  The TREC-8 question answering track report.
    Proceedings of the Eighth Text REtrieval Conference, 77-82, which
    introduced the reciprocal-rank scoring used ever since.
    """
    P_ = C.mat(pred_rank)
    R_ = C.mat(relevant)
    Q = len(P_)
    rr = []
    for q in range(Q):
        rel = set(R_[q])
        v = 0.0
        for k, item in enumerate(P_[q]):
            if item in rel:
                v = 1.0 / (k + 1.0)
                break
        rr.append(v)
    return RichResult(payload={
        "estimate": sum(rr) / Q, "rr": rr,
        "n_hit": sum(1 for v in rr if v > 0.0), "Q": Q,
        "method": "Mean reciprocal rank"})


def cheatsheet():
    return "mrr: Mean reciprocal rank."
