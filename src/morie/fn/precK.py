# morie.fn -- function file (rootcoder007/morie)
"""Precision at k.

Standard information-retrieval ranking metric.  Triage confirmed this
names no owning source: it is the textbook definition used by every
evaluation toolkit, and no citation is manufactured for it here.
"""


from ._richresult import RichResult, with_describe_pointer

__all__ = ["precision_at_k"]


def precision_at_k(pred_rank, relevant, k):
    """Fraction of the top-k retrieved items that are relevant,

        P@k = |rel intersect top-k| / k.

    The denominator is k itself, not the number of relevant items, so
    a short result list is penalized: retrieving three relevant items
    out of ten asked for scores 0.3 even if only three exist.

    Parameters
    ----------
    pred_rank : sequence of item ids, best first.
    relevant : collection of relevant item ids.
    k : int, cutoff rank.

    Returns
    -------
    RichResult with keys estimate, hits, k, n_relevant, method.
    """
    kk = int(k)
    if kk <= 0:
        raise ValueError("k must be positive")
    rel = set(relevant)
    top = list(pred_rank)[:kk]
    hits = sum(1 for t in top if t in rel)
    return with_describe_pointer(RichResult(payload={
        "estimate": float(hits) / kk, "hits": hits, "k": kk,
        "n_relevant": len(rel), "method": "precision at k",
    }), "precK")


def cheatsheet():
    return "precK: Precision@k"


# compact alias per ledger/NAMING.md
precisionk = precision_at_k
