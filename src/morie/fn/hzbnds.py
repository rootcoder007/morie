# morie.fn -- function file (rootcoder007/morie)
"""Horowitz-Manski bounds and the contrast against the MAR point estimate.

Horowitz, J. L. and Manski, C. F. (2000), "Nonparametric analysis of
randomized experiments with missing covariate and outcome data",
Journal of the American Statistical Association 95(449):77-84,
doi:10.1080/01621459.2000.10473902; Manski, C. F. (2003), Partial
Identification of Probability Distributions, Springer.

Missing at random makes E[Y | R = 0] = E[Y | R = 1], which point
identifies E[Y] at the complete-case mean.  Dropping that assumption
leaves only the support restriction y_min <= Y <= y_max, and the
identified set becomes

    [ m p + y_min (1 - p) ,  m p + y_max (1 - p) ],
    m = E[Y | R = 1],  p = P(R = 1).

This function reports both, plus the two contrasts (how far the MAR
answer sits from each end of the identified set), which is the
quantity a sensitivity analysis actually wants: it is the amount of
departure from MAR the data cannot rule out.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["horowitz_manski_bounds"]


def horowitz_manski_bounds(y, R, y_min, y_max):
    """Bounds under missing data with the MAR point estimate for contrast.

    Parameters
    ----------
    y : array-like
        Outcomes; entries at which R = 0 are never read.
    R : array-like
        Response indicator, 1 observed and 0 missing.
    y_min, y_max : float
        A priori support limits.

    Returns
    -------
    estimate : the MAR (complete-case) point estimate
    lower, upper, width : the worst-case identified set
    contrast_lower, contrast_upper : MAR estimate minus each end
    """
    yy = core.vec(y)
    rr = core.vec(R)
    n = len(yy)
    if n == 0:
        raise ValueError("horowitz_manski_bounds: y is empty")
    if len(rr) != n:
        raise ValueError("horowitz_manski_bounds: y and R have different lengths")
    lo = float(y_min)
    hi = float(y_max)
    if not (hi >= lo):
        raise ValueError("horowitz_manski_bounds: y_max must be at least y_min")
    nobs = 0
    s = 0.0
    for i in range(n):
        if rr[i] != 0.0 and rr[i] != 1.0:
            raise ValueError("horowitz_manski_bounds: R must be 0 or 1")
        if rr[i] == 1.0:
            nobs += 1
            s += yy[i]
    if nobs == 0:
        raise ValueError("horowitz_manski_bounds: no observed outcome, the MAR estimate is undefined")
    p = nobs / n
    m = s / nobs
    lower = m * p + lo * (1.0 - p)
    upper = m * p + hi * (1.0 - p)
    return RichResult(
        title="Horowitz-Manski bounds with the MAR contrast",
        summary_lines=[("n", n), ("observed", nobs)],
        payload={
            "estimate": m,
            "mar_estimate": m,
            "lower": lower,
            "upper": upper,
            "width": upper - lower,
            "contrast_lower": m - lower,
            "contrast_upper": upper - m,
            "p_observed": p,
            "n": n,
            "n_observed": nobs,
            "method": "Horowitz-Manski (2000) identified set vs the MAR point estimate",
        },
    )


def cheatsheet():
    return "hzbnds: Horowitz-Manski bounds under missing data"
