# morie.fn -- function file (rootcoder007/morie)
"""NDCG@k for a predicted ranking (alias of :mod:`alndcg`)."""

from .alndcg import alammar_ndcg_at_k

__all__ = ["ndcg"]


def ndcg(pred_rank, relevant, k):
    """Normalised discounted cumulative gain at ``k``.

    This module is an ALIAS.  The gain arithmetic is implemented once,
    in ``alndcg.alammar_ndcg_at_k``; this entry point converts a
    *ranking of items* plus a *relevance lookup* into the graded
    relevance sequence that routine expects, then delegates.

        DCG@k  = sum_{i=1..k} (2^rel_i - 1) / log2(i + 1)
        NDCG@k = DCG@k / IDCG@k

    where ``IDCG@k`` re-sorts the same relevances descending.  An
    all-zero relevance list has ``IDCG = 0``; NDCG is then undefined and
    is refused rather than reported as a perfect 1.

    Parameters
    ----------
    pred_rank : sequence
        Items in predicted order, best first.
    relevant : mapping or container
        If a mapping, ``relevant[item]`` is the graded relevance of
        ``item`` (absent items score 0).  Otherwise membership is used
        as binary relevance.
    k : int
        Cut-off.

    Returns
    -------
    RichResult
        ``estimate`` (NDCG@k), ``dcg``, ``idcg``, ``k``, ``n``.

    References
    ----------
    Jarvelin, K. and Kekalainen, J. (2002), "Cumulated gain-based
    evaluation of IR techniques", ACM Transactions on Information
    Systems 20(4), 422-446, doi:10.1145/582415.582418.
    """
    items = list(pred_rank)
    if not items:
        raise ValueError("pred_rank is empty")
    if hasattr(relevant, "get"):
        rel = [float(relevant.get(it, 0.0)) for it in items]
    else:
        member = list(relevant)
        rel = [1.0 if it in member else 0.0 for it in items]
    return alammar_ndcg_at_k(rel, k)


def cheatsheet():
    return "ndcg: NDCG@k for a predicted ranking (alias of alndcg)"
