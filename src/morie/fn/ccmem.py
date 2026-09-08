# morie.fn -- function file (rootcoder007/morie)
"""Cross-classified membership weight matrix."""

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["cross_classified_membership"]


def cross_classified_membership(y, cluster1, cluster2=None, weights=None):
    """
    Cross-classified membership weight matrix

    Formula: W_ij = 1/n_j if i in cluster j else 0; rows sum to 1

    A unit belonging to several higher-level units at once -- pupils in
    both a school and a neighbourhood -- contributes to each through a
    membership weight, and those weights must sum to one per unit or the
    random-effect variance is rescaled without anyone noticing.  With
    two classifications and no explicit weights each contributes 1/2.

    Parameters
    ----------
    y : array-like
        Response, length n.
    cluster1 : array-like
        First classification label per unit.
    cluster2 : array-like or None
        Second classification; None gives a single classification.
    weights : array-like or None
        Membership weight per classification, recycled over units;
        None splits equally and normalises.

    Returns
    -------
    result : dict
        Keys: estimate (mean row sum, exactly 1), W, row_sums, levels1,
        levels2, n_units, n_levels.

    References
    ----------
    Goldstein (1994), Multilevel cross-classified models, Sociological
    Methods & Research 22(3):364-375.
    Browne, Goldstein & Rasbash (2001), Statistical Modelling
    1(2):103-124.
    """
    yv = core.vec(y)
    n = len(yv)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    c1 = list(cluster1)
    if len(c1) != n:
        raise ValueError("y and cluster1 must have the same length")
    cls = [c1]
    if cluster2 is not None:
        c2 = list(cluster2)
        if len(c2) != n:
            raise ValueError("y and cluster2 must have the same length")
        cls.append(c2)
    C = len(cls)
    if weights is None:
        wc = [1.0 / C] * C
    else:
        wc = core.vec(weights)
        if len(wc) != C:
            raise ValueError("weights must have one entry per classification")
        s = sum(wc)
        if s <= 0.0:
            raise ValueError("classification weights must sum to a positive value")
        wc = [v / s for v in wc]
    levels = []
    for k, cl in enumerate(cls):
        seen = []
        for v in cl:
            if v not in seen:
                seen.append(v)
        levels.append(seen)
    cols = sum(len(L) for L in levels)
    W = [[0.0] * cols for _ in range(n)]
    off = 0
    for k, cl in enumerate(cls):
        for j, lev in enumerate(levels[k]):
            for i in range(n):
                if cl[i] == lev:
                    W[i][off + j] = wc[k]
        off += len(levels[k])
    rs = [sum(r) for r in W]
    return RichResult(payload={
        "estimate": sum(rs) / n,
        "W": W,
        "row_sums": rs,
        "levels1": [float(len(levels[0]))],
        "levels2": [float(len(levels[1]))] if C > 1 else [0.0],
        "n_units": n,
        "n_levels": cols,
        "method": "cross-classified membership weight matrix",
    })


def cheatsheet():
    return "ccmem: cross-classified membership weight matrix"


# compact alias per ledger/NAMING.md
crossclassifiedmembership = cross_classified_membership
