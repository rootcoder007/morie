# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""NDCG@k (Jarvelin and Kekalainen 2002; Alammar Ch 8)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["alammar_ndcg_at_k"]


def alammar_ndcg_at_k(relevances, k):
    """DCG@k = sum (2^rel_i - 1)/log2(i + 1); NDCG = DCG / IDCG.

    IDCG re-sorts THE SAME relevances descending; an all-zero list has
    IDCG 0 and NDCG undefined, which is refused rather than declared
    perfect.

    Examples
    --------
    >>> round(alammar_ndcg_at_k([3, 2, 3, 0, 1, 2], 6)["estimate"], 6)
    0.948811
    """
    r = np.atleast_1d(np.asarray(relevances, dtype=float))
    k = int(k)
    if k < 1:
        raise ValueError("k must be positive.")
    if np.any(r < 0):
        raise ValueError("graded relevances must be non-negative.")

    def dcg(v):
        v = v[:k]
        return float(sum((2.0 ** x - 1) / np.log2(i + 2)
                         for i, x in enumerate(v)))

    got = dcg(r)
    ideal = dcg(np.sort(r)[::-1])
    if ideal == 0:
        raise ValueError(
            "every relevance is 0, so IDCG is 0 and NDCG is undefined; "
            "declaring the ranking perfect there would be a lie.")
    return RichResult(payload={
        "estimate": got / ideal, "dcg": got, "idcg": ideal, "k": k,
        "n": len(r),
        "method": "NDCG@k (Jarvelin and Kekalainen 2002)"})


def cheatsheet():
    return "alndcg: DCG over ideal DCG, all-zero relevances refused"


# compact alias per ledger/NAMING.md
alammarndcgatk = alammar_ndcg_at_k
