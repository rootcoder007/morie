# morie.fn -- function file (rootcoder007/morie)
"""Begg-Mazumdar rank-correlation test for funnel-plot asymmetry."""

from __future__ import annotations

import math

from . import _stats_core as stats
from . import _t4core as T

from ._richresult import RichResult

__all__ = ["ma_begg_test"]


def ma_begg_test(yi, vi):
    """Begg and Mazumdar's rank correlation test for publication bias.

    The effects are standardised so that under the null they are
    independent of their variances:

        ``w_i = 1/v_i``,  ``theta = sum w_i y_i / sum w_i``,
        ``vb = 1 / sum w_i``,  ``v*_i = v_i - vb``,
        ``y*_i = (y_i - theta) / sqrt(v*_i)``

    and the statistic is Kendall's tau between ``y*`` and ``v``, with the
    two-sided normal-approximation p-value.

    Subtracting ``vb`` -- the variance of the fixed-effect summary -- is
    the whole point of the standardisation: without it ``y_i - theta``
    and ``v_i`` are correlated under the null and the test rejects far
    too often.  The test is famously underpowered for small ``k``; the
    p-value here is the asymptotic one, not the exact permutation
    p-value, because the batch forbids resampling.

    Parameters
    ----------
    yi : array-like
        Observed effect sizes.
    vi : array-like
        Their sampling variances (not standard errors).

    Returns
    -------
    RichResult
        ``tau``, ``statistic`` (z), ``p_value``, ``n``, ``method``.

    References
    ----------
    Begg and Mazumdar (1994), Operating characteristics of a rank
    correlation test for publication bias, Biometrics 50:1088-1101.
    Paywalled; the coded form was read from Viechtbauer's ``metafor``
    package, R/ranktest.r (tarball metafor_5.0-1 fetched from CRAN),
    which is the reference implementation and computes exactly
    ``vi.star <- vi - vb; yi.star <- (yi - theta)/sqrt(vi.star)`` then
    ``cor.test(yi.star, vi, method = "kendall")``.
    """
    yi = T.vec(yi)
    vi = T.vec(vi)
    k = len(yi)
    if len(vi) != k:
        raise ValueError("yi and vi must be the same length")
    if k < 3:
        raise ValueError("need at least 3 studies")
    if any(v <= 0 for v in vi):
        raise ValueError("sampling variances must be positive")
    w = [1.0 / v for v in vi]
    sw = sum(w)
    theta = sum(w[i] * yi[i] for i in range(k)) / sw
    vb = 1.0 / sw
    vstar = [vi[i] - vb for i in range(k)]
    if any(v <= 0 for v in vstar):
        raise ValueError("vi - 1/sum(1/vi) must be positive for every study")
    ystar = [(yi[i] - theta) / math.sqrt(vstar[i]) for i in range(k)]
    tau, z = T.kendalltaub(ystar, vi)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z))) if z == z else float("nan")
    return RichResult(
        payload={
            "tau": float(tau),
            "statistic": float(z),
            "p_value": float(p),
            "n": int(k),
            "method": "Begg-Mazumdar rank correlation test",
        }
    )


def cheatsheet():
    return "ma_begg_test(yi, vi): Kendall tau of standardised effect vs variance."


# compact alias per ledger/NAMING.md
mabeggtest = ma_begg_test
