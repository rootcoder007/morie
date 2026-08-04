# morie.fn -- function file (rootcoder007/morie)
"""Hit rate at k.

Standard information-retrieval ranking metric.  Triage confirmed this
names no owning source: it is the textbook definition used by every
evaluation toolkit, and no citation is manufactured for it here.
"""


from ._richresult import RichResult, with_describe_pointer

__all__ = ["hits_at_k"]


def hits_at_k(pred_rank, relevant, k):
    """Whether any relevant item appears in the top k: 1 if the cut
    contains at least one hit, 0 otherwise.

    This is the coarsest of the cutoff metrics -- it ignores how many
    relevant items were found and where they sat -- which is exactly
    why it is used for "did the user get anything useful at all"
    questions.

    Parameters
    ----------
    pred_rank : sequence of item ids, best first.
    relevant : collection of relevant item ids.
    k : int, cutoff rank.

    Returns
    -------
    RichResult with keys estimate, hits, hit, k, n_relevant, method.
    """
    kk = int(k)
    if kk <= 0:
        raise ValueError("k must be positive")
    rel = set(relevant)
    top = list(pred_rank)[:kk]
    hits = sum(1 for t in top if t in rel)
    return with_describe_pointer(RichResult(payload={
        "estimate": 1.0 if hits else 0.0, "hits": hits,
        "hit": bool(hits), "k": kk, "n_relevant": len(rel),
        "method": "hit rate at k",
    }), "hitsR")


def cheatsheet():
    return "hitsR: HitRate@k"


# compact alias per ledger/NAMING.md
hitsk = hits_at_k
