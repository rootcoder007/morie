# morie.fn -- function file (rootcoder007/morie)
"""Rosenbaum sensitivity bounds for the signed-rank test."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["rosenbaum_bound_signed"]


def rosenbaum_bound_signed(pairs, Gamma=1.0):
    """
    Rosenbaum sensitivity bounds, Wilcoxon signed-rank

    Formula: vary Gamma; compute the upper p-value

    Under a bias of at most Gamma in the odds of treatment within a
    matched pair, the null distribution of the signed-rank statistic is
    bounded by the one where each pair contributes its rank with
    probability p+ = Gamma/(1+Gamma).  The upper p-value uses the normal
    approximation with mean p+ sum(q) and variance p+(1-p+) sum(q^2).
    At Gamma = 1 that reduces to the ordinary Wilcoxon signed-rank test,
    which is the reference this is checked against.

    Parameters
    ----------
    pairs : array-like
        Within-pair differences (treated minus control).  Exact zeros
        are dropped, as in the untransformed signed-rank test.
    Gamma : float
        Sensitivity parameter, at least 1.

    Returns
    -------
    result : dict
        Keys: estimate (upper p-value), p_upper, p_lower, W, mu_plus,
        sigma_plus, z_upper, n_pairs, Gamma.

    References
    ----------
    Rosenbaum (2002), Observational Studies, 2nd ed., Springer,
    section 4.3.
    """
    d = [v for v in core.vec(pairs) if v != 0.0]
    n = len(d)
    if n == 0:
        raise ValueError("empty input: no non-zero pair differences")
    G = float(Gamma)
    if G < 1.0:
        raise ValueError("Gamma must be at least 1")
    ranks = core.rank_avg([abs(v) for v in d])
    W = sum(ranks[i] for i in range(n) if d[i] > 0.0)
    pp = G / (1.0 + G)
    pm = 1.0 / (1.0 + G)
    sq = sum(ranks)
    sq2 = sum(v * v for v in ranks)
    mu_p = pp * sq
    sd_p = math.sqrt(pp * (1.0 - pp) * sq2)
    mu_m = pm * sq
    sd_m = math.sqrt(pm * (1.0 - pm) * sq2)
    z_up = (W - mu_p) / sd_p if sd_p > 0.0 else float("nan")
    z_lo = (W - mu_m) / sd_m if sd_m > 0.0 else float("nan")
    p_up = 1.0 - core.pnorm(z_up)
    p_lo = 1.0 - core.pnorm(z_lo)
    return RichResult(payload={
        "estimate": p_up,
        "p_upper": p_up,
        "p_lower": p_lo,
        "W": W,
        "mu_plus": mu_p,
        "sigma_plus": sd_p,
        "z_upper": z_up,
        "n_pairs": n,
        "Gamma": G,
        "method": "Rosenbaum sensitivity bounds, Wilcoxon signed-rank",
    })


def cheatsheet():
    return "cnsRos: Rosenbaum sensitivity bounds (signed-rank)"


# compact alias per ledger/NAMING.md
rosenbaumboundsigned = rosenbaum_bound_signed
