# morie.fn -- function file (rootcoder007/morie)
"""Recall at k.

Standard information-retrieval ranking metric.  Triage confirmed this
names no owning source: it is the textbook definition used by every
evaluation toolkit, and no citation is manufactured for it here.
"""


from ._richresult import RichResult, with_describe_pointer

__all__ = ["recall_at_k"]


def recall_at_k(pred_rank, relevant, k):
    """Fraction of the relevant items that appear in the top k,

        R@k = |rel intersect top-k| / |rel|.

    The denominator is the number of relevant items, so unlike
    precision this cannot be raised by asking for a longer list than
    there are relevant items -- it saturates at 1.

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
    if not rel:
        raise ValueError("recall is undefined with no relevant items")
    top = list(pred_rank)[:kk]
    hits = sum(1 for t in top if t in rel)
    return with_describe_pointer(RichResult(payload={
        "estimate": float(hits) / len(rel), "hits": hits, "k": kk,
        "n_relevant": len(rel), "method": "recall at k",
    }), "colMet")


def cheatsheet():
    return "colMet: Recall@k"


# compact alias per ledger/NAMING.md
recallk = recall_at_k
