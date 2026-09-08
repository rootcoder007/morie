# morie.fn -- function file (rootcoder007/morie)
"""Variance of Hedges' g in a within-subject design."""

import math

from ._richresult import RichResult

__all__ = ["ma_smd_var_correlated_designs"]


def ma_smd_var_correlated_designs(g, n, rho):
    """The variance of a standardised effect when the two scores are paired.

    Treating a pre-post or crossover design as if the two arms were
    independent inflates the variance and, in a meta-analysis, silently
    down-weights exactly the designs that carry the most information.  The
    correlation between the paired measurements enters the variance
    directly: at ``rho = 1`` the sampling variance of the raw difference
    vanishes and only the term from estimating the standardiser survives.

    Formula: ``V_g = J^2 (2(1 - rho)/n + g^2/(2(n-1)))`` with the small-
    sample correction ``J = 1 - 3/(4(n-1) - 1)`` -- Morris (2008)
    eq. (7)-(8); Morris & DeShon (2002).

    Parameters
    ----------
    g : float
        The corrected standardised mean difference.
    n : int
        Number of subjects; ``n >= 2``.
    rho : float
        Correlation between the paired measurements, in ``[-1, 1]``.

    Returns
    -------
    RichResult
        ``var_g``, ``se``, ``J``, ``n``, ``rho``.

    References
    ----------
    Morris, S. B. (2008).  Estimating effect sizes from
    pretest-posttest-control group designs.  Organizational Research
    Methods 11(2):364-386.  doi:10.1177/1094428106291059.
    """
    nn = float(n)
    r = float(rho)
    if nn < 2.0:
        raise ValueError("n must be at least two")
    if r < -1.0 or r > 1.0:
        raise ValueError("rho must lie in [-1, 1]")
    df = nn - 1.0
    J = 1.0 - 3.0 / (4.0 * df - 1.0)
    gg = float(g)
    v = J * J * (2.0 * (1.0 - r) / nn + gg * gg / (2.0 * df))
    return RichResult(payload={
        "var_g": v, "se": math.sqrt(v), "J": J, "n": nn, "rho": r,
        "method": "Variance of Hedges' g for a correlated design"})


def cheatsheet():
    return "marba: variance of Hedges' g in a within-subject design"
