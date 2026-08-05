# morie.fn -- function file (rootcoder007/morie)
"""Design effect of a weighted, clustered sample (Kish)."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["design_effect"]


def design_effect(y, weights=None, cluster=None):
    """
    Design effect of unequal weighting and clustering

    Formula: DEFF = Var_complex / Var_SRS

    Decomposed the way Kish does it, into the unequal-weighting effect
    DEFF_w = n sum w_i^2 / (sum w_i)^2 and the clustering effect
    DEFF_c = 1 + (m0 - 1) rho, where rho is the one-way ANOVA
    intraclass correlation and m0 the Kish average cluster size.  The
    reported DEFF is their product.  Equal weights give DEFF_w = 1
    exactly, and singleton clusters give DEFF_c = 1 exactly.

    Parameters
    ----------
    y : array-like
        Survey variable, length n.
    weights : array-like or None
        Sampling weights.  None means equal weights.
    cluster : array-like or None
        Cluster (PSU) identifier per unit.  None means no clustering.

    Returns
    -------
    result : dict
        Keys: estimate (DEFF), deff_w, deff_c, rho, m0, n_eff, n.

    References
    ----------
    Kish (1965), Survey Sampling, Wiley, sections 8.2 and 5.4.
    """
    y = core.vec(y)
    n = len(y)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    w = [1.0] * n if weights is None else core.vec(weights)
    if len(w) != n:
        raise ValueError("y and weights must have the same length")
    for v in w:
        if v < 0.0:
            raise ValueError("weights must be non-negative")
    sw = sum(w)
    if sw <= 0.0:
        raise ValueError("weights must not sum to zero")
    deff_w = n * sum(v * v for v in w) / (sw * sw)
    if cluster is None:
        ids = list(range(n))
    else:
        ids = list(cluster)
        if len(ids) != n:
            raise ValueError("y and cluster must have the same length")
    keys = []
    for k in ids:
        if k not in keys:
            keys.append(k)
    groups = [[y[i] for i in range(n) if ids[i] == k] for k in keys]
    a = len(groups)
    sizes = [len(g) for g in groups]
    gm = sum(y) / n
    ssb = sum(sizes[j] * (sum(groups[j]) / sizes[j] - gm) ** 2 for j in range(a))
    ssw = sum((v - sum(g) / len(g)) ** 2 for g in groups for v in g)
    if a > 1 and n > a:
        msb = ssb / (a - 1)
        msw = ssw / (n - a)
        m0 = (n - sum(s * s for s in sizes) / float(n)) / (a - 1)
        denom = msb + (m0 - 1.0) * msw
        rho = (msb - msw) / denom if denom != 0.0 else 0.0
    else:
        msb = msw = float("nan")
        m0 = float(n) / a
        rho = 0.0
    deff_c = 1.0 + (m0 - 1.0) * rho
    deff = deff_w * deff_c
    return RichResult(payload={
        "estimate": deff,
        "deff_w": deff_w,
        "deff_c": deff_c,
        "rho": rho,
        "m0": m0,
        "n_eff": n / deff if deff > 0.0 else float("nan"),
        "n": n,
        "method": "Kish design effect: unequal weighting x clustering",
    })


def cheatsheet():
    return "desigeff: Kish design effect of weighting and clustering"


# compact alias per ledger/NAMING.md
designeffect = design_effect
