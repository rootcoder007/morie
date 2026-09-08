# morie.fn -- function file (rootcoder007/morie)
"""Mean average precision at k.

Standard information-retrieval ranking metric.  Triage confirmed this
names no owning source: it is the textbook definition used by every
evaluation toolkit, and no citation is manufactured for it here.

Normalization warning.  Average precision at a cutoff has two
conventions in circulation: divide the summed precisions by
min(|rel|, k), or by |rel|.  The first is used here -- it is the form
that reaches 1 when every one of the top k is relevant -- and the
second is returned alongside as ``ap_over_nrel`` so neither is
silently substituted for the other.
"""


from ._richresult import RichResult, with_describe_pointer

__all__ = ["map_at_k"]


def _ap_at_k(rank, rel, k):
    """Average precision at k for a single ranking: the mean of the
    precisions measured at each rank that holds a relevant item."""
    top = list(rank)[:k]
    hits = 0
    s = 0.0
    for i, t in enumerate(top, start=1):
        if t in rel:
            hits += 1
            s += hits / float(i)
    return s, hits


def map_at_k(pred_rank, relevant, k):
    """Mean average precision at k.

    ``pred_rank`` may be a single ranking with ``relevant`` a single
    collection, or a sequence of rankings with ``relevant`` a matching
    sequence of collections, in which case the average precisions are
    averaged over the queries.

    Parameters
    ----------
    pred_rank : ranking, or sequence of rankings, best first.
    relevant : relevant ids, or sequence of such collections.
    k : int, cutoff rank.

    Returns
    -------
    RichResult with keys estimate, ap, ap_over_nrel, n_queries, k,
    method.
    """
    kk = int(k)
    if kk <= 0:
        raise ValueError("k must be positive")
    ranks = list(pred_rank)
    nested = bool(ranks) and all(
        isinstance(r, (list, tuple)) for r in ranks)
    if nested:
        rels = [set(r) for r in relevant]
        if len(rels) != len(ranks):
            raise ValueError("need one relevant set per ranking")
    else:
        ranks = [ranks]
        rels = [set(relevant)]
    aps = []
    apn = []
    for rk, rl in zip(ranks, rels):
        if not rl:
            raise ValueError("average precision needs relevant items")
        s, _ = _ap_at_k(rk, rl, kk)
        aps.append(s / min(len(rl), kk))
        apn.append(s / len(rl))
    n = len(aps)
    return with_describe_pointer(RichResult(payload={
        "estimate": sum(aps) / n, "ap": aps, "ap_over_nrel": apn,
        "n_queries": n, "k": kk,
        "method": "mean average precision at k",
    }), "mapMet")


def cheatsheet():
    return "mapMet: MAP@k"


# compact alias per ledger/NAMING.md
mapk = map_at_k
