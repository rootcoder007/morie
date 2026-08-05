# morie.fn -- wave2 slice x_2_01 (rootcoder007/morie)
"""IPW estimate with a replicate-weight (jackknife) variance.

The point estimate is the Hajek weighted difference in means between
treated and control units; its variance is taken from the replicate
weights supplied with the survey, which is the standard design-based
route when the sampling design is not recoverable from the microdata:

    v(theta) = ((B - 1) / B) * sum_b (theta_b - theta)^2,

the JK1 (delete-one-group jackknife) form given by Rust and Rao (1996),
"Variance estimation for complex surveys using replication techniques",
Statistical Methods in Medical Research 5(3):283-310,
doi:10.1177/096228029600500305.  Lumley (2010), *Complex Surveys: A
Guide to Analysis Using R*, Wiley, doi:10.1002/9780470580066, and
Lumley and Scott (2017), "Fitting regression models to survey data",
Statistical Science 32(2):265-278, doi:10.1214/16-STS605, describe the
same construction for weighted regression estimators; the scale factor
is exposed so JKn and BRR schemes (factor 1 / B) can be used instead.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ipw_with_replicate"]


def _hajek_diff(yv, d, w):
    s1 = 0.0
    s0 = 0.0
    t1 = 0.0
    t0 = 0.0
    for i in range(len(yv)):
        if d[i] == 1.0:
            s1 += w[i]
            t1 += w[i] * yv[i]
        else:
            s0 += w[i]
            t0 += w[i] * yv[i]
    if s1 <= 0 or s0 <= 0:
        raise ValueError("ipw_with_replicate: a replicate leaves one arm with no weight")
    return t1 / s1 - t0 / s0


def ipw_with_replicate(y, D, w, replicate_weights, scale=None):
    """Weighted treatment effect with a replicate-weight variance.

    Parameters
    ----------
    y : array-like
        Outcome.
    D : array-like
        Treatment indicator, 0 or 1.
    w : array-like
        Full-sample analysis weights (design weight times IP weight).
    replicate_weights : array-like
        n x B matrix of replicate weights, one column per replicate.
    scale : float or None
        Multiplier on the sum of squared deviations.  None uses the JK1
        value (B - 1) / B.
    """
    yv = core.vec(y)
    n = len(yv)
    if n == 0:
        raise ValueError("ipw_with_replicate: y is empty")
    d = core.vec(D)
    wv = core.vec(w) if w is not None else [1.0] * n
    if len(d) != n or len(wv) != n:
        raise ValueError("ipw_with_replicate: y, D and w have different lengths")
    for v in d:
        if v not in (0.0, 1.0):
            raise ValueError("ipw_with_replicate: D must be 0 or 1")
    R = core.mat(replicate_weights)
    if len(R) != n:
        raise ValueError("ipw_with_replicate: replicate_weights must have one row per observation")
    B = len(R[0]) if R else 0
    if B < 2:
        raise ValueError("ipw_with_replicate: need at least two replicates")
    theta = _hajek_diff(yv, d, wv)
    reps = []
    for b in range(B):
        reps.append(_hajek_diff(yv, d, [R[i][b] for i in range(n)]))
    fac = (B - 1.0) / B if scale is None else float(scale)
    var = fac * sum((r - theta) ** 2 for r in reps)
    return RichResult(
        title="IPW with replicate-weight variance",
        summary_lines=[("n", n), ("replicates", B), ("estimate", theta), ("se", math.sqrt(var))],
        payload={
            "estimate": theta,
            "se": math.sqrt(var),
            "variance": var,
            "scale": fac,
            "replicate_estimates": reps,
            "rep_mean": sum(reps) / B,
            "n_replicates": B,
            "n": n,
            "method": "Hajek weighted difference; v = ((B-1)/B) sum_b (theta_b - theta)^2, Rust & Rao (1996) JK1",
        },
    )


def cheatsheet():
    return "ipferd: IPW with replicate-weight variance"


# compact alias per ledger/NAMING.md
ipwreplicate = ipw_with_replicate
