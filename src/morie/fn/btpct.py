# morie.fn -- slice s03 (rootcoder007/morie)
"""Percentile bootstrap confidence interval.

Source consulted: Efron, B. (1979).  Bootstrap methods: another look at
the jackknife.  *The Annals of Statistics* 7(1), 1-26, and Efron, B. and
Tibshirani, R. J. (1993).  *An Introduction to the Bootstrap*, Chapman
and Hall, chapter 13.  The percentile interval is simply the alpha/2 and
1 - alpha/2 quantiles of the bootstrap replicates,

    [ thetahat*_(alpha/2) , thetahat*_(1 - alpha/2) ]

Neither source was retrievable here as a full text; the interval is
quoted in its standard published form.

The percentile interval is *not* second-order accurate when the
replicate distribution is skewed or biased; the bias-corrected
acceleration-free interval (BC, Efron and Tibshirani chapter 14) is
returned alongside as ``bc_lo``/``bc_hi`` so the discrepancy is visible.
Quantiles use R's type 7 in both arms.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["boot_percentile_ci"]


def boot_percentile_ci(theta_b, alpha=0.05, theta_hat=None):
    """Percentile (and bias-corrected) bootstrap interval.

    Returns
    -------
    lo, hi   : the percentile endpoints
    estimate : the interval width
    bc_lo, bc_hi : the bias-corrected endpoints
    z0       : the bias correction
    """
    v = k.vec(theta_b)
    n = len(v)
    a = float(alpha)
    lo = k.quantile7(v, a / 2.0)
    hi = k.quantile7(v, 1.0 - a / 2.0)
    th = float(theta_hat) if theta_hat is not None else k.mean(v)
    cnt = 0.0
    for x in v:
        if x < th:
            cnt += 1.0
    p = cnt / n if n else 0.5
    if p <= 0.0:
        p = 0.5 / n
    if p >= 1.0:
        p = 1.0 - 0.5 / n
    z0 = k.qnorm(p)
    za = k.qnorm(a / 2.0)
    zb = k.qnorm(1.0 - a / 2.0)
    a1 = k.pnorm(2.0 * z0 + za)
    a2 = k.pnorm(2.0 * z0 + zb)
    return RichResult(
        title="Percentile bootstrap interval",
        summary_lines=[("lo", lo), ("hi", hi)],
        payload={
            "lo": lo,
            "hi": hi,
            "estimate": hi - lo,
            "bc_lo": k.quantile7(v, a1),
            "bc_hi": k.quantile7(v, a2),
            "z0": z0,
            "n": n,
            "method": "Efron (1979) percentile bootstrap interval, with the bias-corrected variant",
        },
    )


def cheatsheet():
    return "btpct: Percentile bootstrap CI"
