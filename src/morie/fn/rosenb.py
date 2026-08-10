# morie.fn -- function file (rootcoder007/morie)
"""Rosenbaum sensitivity bounds over a grid of Gamma (wraps cnsRos)."""

from .cnsRos import rosenbaum_bound_signed
from ._richresult import RichResult

__all__ = ["rosenb", "rosenbaum_bounds"]


def rosenb(matched_pairs, Gamma_grid=(1.0, 1.5, 2.0, 3.0), alpha=0.05):
    """Rosenbaum bounds for matched studies, tabulated over Gamma.

    For each sensitivity parameter ``Gamma`` (the largest factor by
    which two units matched on covariates may differ in their odds of
    treatment), the null distribution of the Wilcoxon signed-rank
    statistic is bracketed by two extreme distributions in which each
    pair contributes its rank with probability ``p+ = Gamma/(1+Gamma)``
    (upper) or ``p- = 1/(1+Gamma)`` (lower).  This routine evaluates the
    per-Gamma bounds by calling
    :func:`morie.fn.cnsRos.rosenbaum_bound_signed` -- the single-Gamma
    machinery already verified against ``stats::wilcox.test`` at
    ``Gamma = 1`` -- and tabulates the p-value intervals, which is how
    Rosenbaum presents the analysis.  ``gamma_critical`` is the largest
    grid value whose upper p-value still falls below ``alpha`` (None if
    already insensitive at the smallest Gamma).

    Parameters
    ----------
    matched_pairs : array-like
        Within-pair differences (treated minus control); zeros dropped.
    Gamma_grid : array-like of float
        Sensitivity parameters, each at least 1.
    alpha : float
        Level used to report ``gamma_critical``.

    Returns
    -------
    RichResult
        ``Gamma``, ``p_upper``, ``p_lower`` (parallel lists),
        ``gamma_critical``, ``n_pairs``, ``W``.

    References
    ----------
    Rosenbaum, P. R. (2002), Observational Studies, 2nd ed., Springer,
    Section 4.3 (signed-rank bounds) and the Gamma-table presentation of
    Chapter 4.  Single-Gamma engine: morie.fn.cnsRos (verified vs
    stats::wilcox.test at Gamma = 1).
    """
    gs = [float(g) for g in Gamma_grid]
    if not gs:
        raise ValueError("rosenb: Gamma_grid is empty")
    pu = []
    pl = []
    W = None
    n_pairs = None
    for g in gs:
        r = rosenbaum_bound_signed(matched_pairs, Gamma=g)
        pu.append(float(r["p_upper"]))
        pl.append(float(r["p_lower"]))
        W = float(r["W"])
        n_pairs = int(r["n_pairs"])
    crit = None
    for g, p in zip(gs, pu):
        if p <= float(alpha):
            crit = g
    return RichResult(payload={
        "Gamma": gs, "p_upper": pu, "p_lower": pl,
        "gamma_critical": crit, "n_pairs": n_pairs, "W": W,
        "alpha": float(alpha),
        "method": "Rosenbaum signed-rank sensitivity bounds over Gamma"})


# stub-era long name, kept as an alias
rosenbaum_bounds = rosenb


def cheatsheet():
    return "rosenb: Rosenbaum bounds over a Gamma grid -- wraps cnsRos"
