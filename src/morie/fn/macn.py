# morie.fn -- function file (rootcoder007/morie)
"""Cochran's Q test for heterogeneity."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['cochranq', 'ma_cochran_q']


def cochranq(yi, vi):
    """Cochran's Q test for heterogeneity.

    Q is a weighted residual sum of squares around the fixed-effect pooled estimate, so it tests whether the studies are consistent with one common effect. Its power is poor with few studies and excessive with many, which is why the statistic is returned alongside the p-value rather than the p-value alone.


    Formula: Q = sum_i w_i (y_i - theta_FE)^2 with w_i = 1/v_i, theta_FE = sum w y / sum w; Q ~ chi2_{k-1}

    Parameters
    ----------
    yi : array-like
        Effect estimates, one per study.
    vi : array-like
        Their sampling variances.

    Returns
    -------
    RichResult
        ``Q``, ``df``, ``p_value``, ``theta_fe``, ``se_fe``, ``weights``, ``k``.

    References
    ----------
    Cochran (1954), The combination of estimates from different
    experiments, Biometrics 10:101-129.  Not held locally; Q = sum w_i
    (y_i - theta_FE)^2 on k-1 degrees of freedom is the standard
    published form and is what metafor's rma() reports as QE.
    """
    y = C.vec(yi); v = C.vec(vi)
    k = len(y)
    if k != len(v):
        raise ValueError("yi and vi must be the same length")
    if any(t <= 0 for t in v):
        raise ValueError("variances must be positive")
    w = [1.0 / t for t in v]
    sw = sum(w)
    th = sum(w[i] * y[i] for i in range(k)) / sw
    Q = sum(w[i] * (y[i] - th) ** 2 for i in range(k))
    df = k - 1
    return RichResult(payload={
        "Q": Q, "df": df,
        "p_value": 1.0 - C.pchisq(Q, df) if df > 0 else float("nan"),
        "theta_fe": th, "se_fe": math.sqrt(1.0 / sw), "weights": w, "k": k,
        "method": "Cochran's Q test for heterogeneity"})


ma_cochran_q = cochranq


def cheatsheet():
    return "macn: Cochran's Q test for heterogeneity."
