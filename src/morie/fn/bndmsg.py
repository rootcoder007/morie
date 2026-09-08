# morie.fn -- function file (rootcoder007/morie)
"""Worst-case bound on a mean under a missing outcome.

Horowitz, J. L. and Manski, C. F. (2000), "Nonparametric analysis of
randomized experiments with missing covariate and outcome data",
Journal of the American Statistical Association 95(449):77-84,
doi:10.1080/01621459.2000.10473902; and Manski, C. F. (2003), Partial
Identification of Probability Distributions, Springer.

With R = 1 when Y is observed and R = 0 when it is not, the law of total
expectation splits the target into an identified and an unidentified
piece,

    E[Y] = E[Y | R = 1] P(R = 1) + E[Y | R = 0] P(R = 0),

and only the second factor of the second term is unknown.  Bounding the
unobserved conditional mean by the a priori support [y_min, y_max] gives
the sharp worst-case interval

    L = E[Y | R = 1] P(R = 1) + y_min P(R = 0)
    U = E[Y | R = 1] P(R = 1) + y_max P(R = 0)

whose width is exactly (y_max - y_min) P(R = 0).  Nothing is assumed
about the missingness mechanism: the interval is the identified set.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["bound_missing_outcome"]


def bound_missing_outcome(y, R, y_min, y_max):
    """Sharp worst-case bounds on E[Y] when Y is missing where R = 0.

    Parameters
    ----------
    y : array-like
        Outcomes.  Entries at which R = 0 are never read.
    R : array-like
        Response indicator, 1 for observed and 0 for missing.
    y_min, y_max : float
        A priori lower and upper limits of the support of Y.

    Returns
    -------
    estimate : the midpoint of the identified interval
    lower, upper, width : the interval itself
    p_observed : the observed fraction
    """
    yy = core.vec(y)
    rr = core.vec(R)
    n = len(yy)
    if n == 0:
        raise ValueError("bound_missing_outcome: y is empty")
    if len(rr) != n:
        raise ValueError("bound_missing_outcome: y and R have different lengths")
    lo = float(y_min)
    hi = float(y_max)
    if not (hi >= lo):
        raise ValueError("bound_missing_outcome: y_max must be at least y_min")
    nobs = 0
    s = 0.0
    for i in range(n):
        if rr[i] != 0.0 and rr[i] != 1.0:
            raise ValueError("bound_missing_outcome: R must be 0 or 1")
        if rr[i] == 1.0:
            nobs += 1
            s += yy[i]
    p = nobs / n
    m = s / nobs if nobs > 0 else 0.0
    lower = m * p + lo * (1.0 - p)
    upper = m * p + hi * (1.0 - p)
    return RichResult(
        title="Horowitz-Manski worst-case bound under a missing outcome",
        summary_lines=[("n", n), ("observed", nobs)],
        payload={
            "estimate": 0.5 * (lower + upper),
            "lower": lower,
            "upper": upper,
            "width": upper - lower,
            "p_observed": p,
            "mean_observed": m,
            "n": n,
            "n_observed": nobs,
            "method": "Horowitz-Manski (2000) worst-case bound; width = (y_max - y_min) P(R = 0)",
        },
    )


def cheatsheet():
    return "bndmsg: Bound under missing outcome"
