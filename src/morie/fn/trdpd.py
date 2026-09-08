# morie.fn -- function file (rootcoder007/morie)
"""Tree-depth saturation diagnostic for NUTS."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["treedepth", "tree_depth_saturation"]


def treedepth(depths, max_depth=10):
    """How often the NUTS trajectory hit the tree-depth cap.

    Saturation is an EFFICIENCY warning, not a validity one, and the
    distinction matters: unlike a divergence, a capped trajectory does
    not bias the draws, it just means the sampler was cut off before
    the U-turn criterion fired and is exploring more slowly than it
    could.  The usual cause is a badly scaled posterior, and raising
    the cap treats the symptom.

    Formula: saturated = #{ i : depth_i >= max_depth } / n;
             leapfrog steps per iteration ~ 2^depth_i

    Parameters
    ----------
    depths : array-like
        Tree depth reached at each iteration, non-negative integers.
    max_depth : int
        The configured maximum tree depth.

    Returns
    -------
    RichResult
        ``saturated`` (fraction), ``n_saturated``, ``mean_depth``,
        ``max_observed``, ``mean_leapfrog``, ``total_leapfrog``,
        ``warn`` (1 when any iteration saturated), ``n``.

    References
    ----------
    Hoffman & Gelman (2014), The No-U-Turn Sampler: adaptively setting
    path lengths in Hamiltonian Monte Carlo, Journal of Machine
    Learning Research 15, 1593-1623, which introduces the doubling
    scheme whose depth this counts; and Betancourt (2017), A conceptual
    introduction to Hamiltonian Monte Carlo, arXiv:1701.02434, for the
    reading of saturation as an efficiency rather than a validity
    problem.  Gelman, Carlin, Stern, Dunson, Vehtari & Rubin (2013),
    Bayesian Data Analysis, 3rd edition, was fetched in full and
    searched; it describes HMC but not the tree-depth diagnostic, so it
    is not cited for this.
    """
    d = C.vec(depths)
    n = len(d)
    if n < 1:
        raise ValueError("at least one iteration is required")
    if any(v < 0 for v in d):
        raise ValueError("tree depths must be non-negative")
    md = int(max_depth)
    if md < 0:
        raise ValueError("max_depth must be non-negative")
    k = sum(1 for v in d if v >= md)
    lf = [2.0 ** v for v in d]
    return RichResult(payload={
        "saturated": k / n, "n_saturated": float(k),
        "mean_depth": sum(d) / n, "max_observed": max(d),
        "mean_leapfrog": sum(lf) / n, "total_leapfrog": sum(lf),
        "warn": 1.0 if k > 0 else 0.0, "n": float(n),
        "method": "NUTS tree-depth saturation diagnostic"})


tree_depth_saturation = treedepth


def cheatsheet():
    return "trdpd: fraction of iterations with depth >= max_depth; efficiency not bias"
