# morie.fn -- function file (rootcoder007/morie)
"""Cluster-sampling design: optimal cluster size under a cost function."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["clusdes", "cluster_design"]


def clusdes(rho, S2, c1, c2, budget):
    """Choose the cluster size that buys the most precision per unit cost.

    The design question, not the estimation one: given that elements in
    the same cluster are correlated (rho) and that visiting a cluster
    costs c1 while measuring an element costs c2, how many elements
    should be taken per cluster?  The answer is the classic
    square-root rule, and the reason it is not "as many as possible" is
    the design effect 1 + (k - 1) rho: correlated elements add cost
    faster than they add information.

    The optimum is reported both exactly (a real number) and rounded to
    the better of its two neighbouring integers, compared on the actual
    achieved variance rather than on the rounding.

    Formula: k_opt = sqrt( c1 (1 - rho) / (c2 rho) );
             m = budget / (c1 + c2 k);
             V(ybar) = (S^2 / (m k)) [1 + (k - 1) rho]

    Parameters
    ----------
    rho : float
        Intraclass correlation, 0 < rho <= 1.
    S2 : float
        Element variance in the population.
    c1 : float
        Cost of adding one cluster.
    c2 : float
        Cost of adding one element within a cluster.
    budget : float
        Total budget.

    Returns
    -------
    RichResult
        ``k_opt``, ``k``, ``m``, ``variance``, ``deff``, ``cost``,
        ``elements``.

    References
    ----------
    Cochran (1977), Sampling Techniques, 3rd edition, Chapter 9, which
    develops the design effect 1 + (k - 1) rho for clusters of size k
    and optimises k against the linear cost function c1 m + c2 m k.
    Chapter 9 was NOT in the scanned excerpt available to this batch,
    so the standard published form is used.
    """
    rho = float(rho)
    S2 = float(S2)
    c1 = float(c1)
    c2 = float(c2)
    budget = float(budget)
    if not 0.0 < rho <= 1.0:
        raise ValueError("rho must satisfy 0 < rho <= 1")
    if S2 <= 0:
        raise ValueError("S2 must be positive")
    if c1 <= 0 or c2 <= 0:
        raise ValueError("costs must be positive")
    if budget <= c1 + c2:
        raise ValueError("the budget cannot buy even one cluster of one")
    kopt = math.sqrt(c1 * (1.0 - rho) / (c2 * rho)) if rho < 1.0 else 1.0

    def V(kk):
        mm = budget / (c1 + c2 * kk)
        if mm <= 0:
            return float("inf"), mm
        return (S2 / (mm * kk)) * (1.0 + (kk - 1.0) * rho), mm

    lo = max(1, int(math.floor(kopt)))
    hi = lo + 1
    vlo = V(lo)[0]
    vhi = V(hi)[0]
    k = lo if vlo <= vhi else hi
    var, m = V(k)
    return RichResult(payload={
        "k_opt": kopt, "k": float(k), "m": m, "variance": var,
        "deff": 1.0 + (k - 1.0) * rho, "cost": m * (c1 + c2 * k),
        "elements": m * k,
        "method": "Optimal cluster size under a linear cost function"})


cluster_design = clusdes


def cheatsheet():
    return "clstrs: k_opt = sqrt(c1(1-rho)/(c2 rho)); deff = 1+(k-1)rho"
