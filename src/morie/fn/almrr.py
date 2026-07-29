# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mean reciprocal rank (Alammar Ch 8)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["alammar_mean_reciprocal_rank"]


def alammar_mean_reciprocal_rank(rankings, relevant_indices):
    """MRR = mean over queries of 1 / rank of the first relevant item.

    ``rankings`` is a list of ranked id lists; ``relevant_indices`` a
    list of relevant-id sets. A query whose relevant items never
    appear contributes 0, and how many did is reported -- averaging
    away missing answers silently is how retrieval demos lie.

    Examples
    --------
    >>> alammar_mean_reciprocal_rank([[3, 1, 2]], [[1]])["estimate"]
    0.5
    """
    if len(rankings) != len(relevant_indices):
        raise ValueError("need one relevant set per ranking.")
    if not rankings:
        raise ValueError("no queries supplied.")
    rrs = []
    missed = 0
    for ranked, rel in zip(rankings, relevant_indices):
        rel = set(rel)
        rr = 0.0
        for pos, item in enumerate(ranked, start=1):
            if item in rel:
                rr = 1.0 / pos
                break
        if rr == 0.0:
            missed += 1
        rrs.append(rr)
    return RichResult(payload={
        "estimate": float(np.mean(rrs)),
        "reciprocal_ranks": rrs, "queries_missed": missed,
        "n": len(rrs),
        "method": "Mean reciprocal rank (Alammar Ch 8)"})


def cheatsheet():
    return "almrr: mean 1/rank-of-first-relevant, misses counted not hidden"
