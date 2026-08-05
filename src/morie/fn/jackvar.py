# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Jackknife variance for survey estimates.

Wolter (2007), *Introduction to Variance Estimation*, 2nd ed.,
Springer, chapter 4, equation (4.2.5): with theta_(r) the estimate
recomputed from the r-th replicate,

    v_J = ((R - 1) / R) sum_r (theta_(r) - theta_hat)^2.

For the delete-one jackknife of an unweighted mean this reduces
algebraically to s^2 / n, the textbook variance of the sample mean,
which is the check the tests apply.  Replicate weights may be supplied
instead, in which case each replicate is whatever the supplied weights
make it.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["jackknife_variance_survey"]


def _wmean(v, w):
    sw = 0.0
    s = 0.0
    for i in range(len(v)):
        sw += w[i]
        s += w[i] * v[i]
    if sw == 0:
        raise ValueError("jackknife_variance_survey: replicate weights sum to zero")
    return s / sw


def jackknife_variance_survey(y, weights=None, replicates=None):
    """Jackknife variance of a weighted mean.

    Parameters
    ----------
    y : array-like
        Observations.
    weights : array-like, optional
        Full-sample weights; equal weights by default.
    replicates : R x n matrix, optional
        Replicate weight sets; the delete-one jackknife by default.
    """
    v = core.vec(y)
    n = len(v)
    if n < 2:
        raise ValueError("jackknife_variance_survey: need at least two observations")
    w = [1.0] * n if weights is None else core.vec(weights)
    if len(w) != n:
        raise ValueError("jackknife_variance_survey: weights and y have different lengths")
    for x in w:
        if x < 0:
            raise ValueError("jackknife_variance_survey: weights must be non-negative")
    if replicates is None:
        rep = [[0.0 if i == r else w[i] for i in range(n)] for r in range(n)]
    else:
        rep = core.mat(replicates)
        for row in rep:
            if len(row) != n:
                raise ValueError("jackknife_variance_survey: replicates must have n columns")
    R = len(rep)
    if R < 2:
        raise ValueError("jackknife_variance_survey: need at least two replicates")
    theta = _wmean(v, w)
    th = [_wmean(v, rep[r]) for r in range(R)]
    ss = 0.0
    for t in th:
        ss += (t - theta) ** 2
    var = (R - 1.0) / R * ss
    return RichResult(
        title="Jackknife variance",
        summary_lines=[("n", n), ("replicates", R)],
        payload={
            "estimate": var,
            "variance": var,
            "theta": theta,
            "theta_replicates": th,
            "se": var ** 0.5,
            "n": n,
            "method": "v_J = ((R-1)/R) sum_r (theta_r - theta)^2, Wolter (2007) eq. (4.2.5)",
        },
    )


def cheatsheet():
    return "jackvar: jackknife variance for survey estimates"
