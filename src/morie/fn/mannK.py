# morie.fn -- function file (rootcoder007/morie)
"""Mann-Kendall trend test."""

from __future__ import annotations

import math

from . import _stats_core as stats
from . import _t4core as T

from ._richresult import RichResult

__all__ = ["mann_kendall"]


def mann_kendall(x, continuity=True):
    """Mann-Kendall rank test for monotone trend.

    Formula: ``S = sum_{i<j} sign(x_j - x_i)``, with null variance

        ``var(S) = [n(n-1)(2n+5) - sum_t t(t-1)(2t+5)] / 18``

    where ``t`` runs over the multiplicities of the tied values, and

        ``z = sign(S) (|S| - 1) / sqrt(var S)``

    under the continuity correction.  Kendall's tau uses the tie-adjusted
    denominator ``D = sqrt(n(n-1)/2 - sum t(t-1)/2) sqrt(n(n-1)/2)``,
    which is tau-b against an untied time index.

    The continuity correction is what keeps the test conservative for
    short series; it is applied to ``|S|`` rather than to ``z``, so a
    zero ``S`` gives ``z = 0`` and not a sign flip.

    Parameters
    ----------
    x : array-like
        Series in time order.
    continuity : bool
        Apply the ``|S| - 1`` continuity correction.

    Returns
    -------
    RichResult
        ``statistic`` (z), ``p_value``, ``S``, ``varS``, ``tau``, ``n``,
        ``method``.

    References
    ----------
    Mann (1945), Nonparametric tests against trend, Econometrica
    13:245-259; Kendall (1975), Rank Correlation Methods.  Both
    paywalled; the coded form was read from Pohlert's CRAN package
    ``trend`` (R/mk.test.R and R/utilfn.R, source tarball trend_1.1.7
    fetched from CRAN), whose ``.varmk`` and ``.Dfn`` give the tie
    corrections verbatim.
    """
    x = T.vec(x)
    n = len(x)
    if n < 3:
        raise ValueError("need at least 3 observations")
    s = 0.0
    for j in range(n):
        for i in range(j + 1):
            d = x[j] - x[i]
            s += 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
    tt = T.tiecounts(x)
    tadjs = sum(t * (t - 1.0) * (2.0 * t + 5.0) for t in tt)
    vars_ = (n * (n - 1.0) * (2.0 * n + 5.0) - tadjs) / 18.0
    tadjd = sum(t * (t - 1.0) for t in tt)
    den = math.sqrt(0.5 * n * (n - 1.0) - 0.5 * tadjd) * math.sqrt(0.5 * n * (n - 1.0))
    tau = s / den if den > 0 else float("nan")
    if vars_ <= 0:
        z = float("nan")
    elif continuity:
        sg = 1.0 if s > 0 else (-1.0 if s < 0 else 0.0)
        z = sg * (abs(s) - 1.0) / math.sqrt(vars_)
    else:
        z = s / math.sqrt(vars_)
    p = 2.0 * min(0.5, 1.0 - stats.norm.cdf(abs(z)))
    return RichResult(
        payload={
            "statistic": float(z),
            "p_value": float(p),
            "S": float(s),
            "varS": float(vars_),
            "tau": float(tau),
            "n": int(n),
            "method": "Mann-Kendall trend test",
        }
    )


def cheatsheet():
    return "mann_kendall(x): S = sum sign(x_j - x_i), z from tie-adjusted var(S)."


# compact alias per ledger/NAMING.md
mannkendall = mann_kendall
