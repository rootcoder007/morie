# morie.fn -- slice k04 (rootcoder007/morie)
"""Kendall (1938) rank-correlation test for monotone trend.

Source: Kendall, M. G. (1938).  A new measure of rank correlation.
*Biometrika* 30, 81-93.  The 1938 paper is paywalled here; the measure
is quoted in its standard published form, which is unambiguous::

    tau = (P - Q) / (P + Q)

with P the number of concordant and Q the number of discordant pairs.
Applied with the first argument being time, this is the Mann-Kendall
trend test: the Mann-Kendall S statistic is S = P - Q, and the normal
approximation with the usual tie correction

    Var(S) = [ n(n-1)(2n+5) - sum_g t_g (t_g - 1)(2 t_g + 5) ] / 18

gives the two-sided p-value.

The tau, S and p-value are taken from ``_stats_core.kendalltau``, which
already implements tau-b with the exact null distribution for small n;
re-deriving them here would be a second copy to keep in step.

The previous body of this module was a one-sample Kolmogorov-Smirnov
test against a fitted normal, pasted by the stub generator.  Deleted.
"""

from __future__ import annotations

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["linear_trend"]


def linear_trend(t, x):
    """Kendall rank-correlation test for trend in ``x`` against ``t``.

    Parameters
    ----------
    t : array-like, shape (n,)
        Ordering variable, normally time.
    x : array-like, shape (n,)
        Series to test.

    Returns
    -------
    RichResult
        keys: ``tau``, ``S`` (Mann-Kendall concordance excess),
        ``p_value``, ``n_concordant``, ``n_discordant``, ``n``,
        ``method``.
    """
    t = np.asarray(t, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    n = int(t.size)
    if x.size != n:
        raise ValueError("t and x must have the same length")
    if n < 3:
        raise ValueError("need n >= 3")

    conc = 0
    disc = 0
    for i in range(n - 1):
        dt = t[i + 1 :] - t[i]
        dx = x[i + 1 :] - x[i]
        prod = dt * dx
        conc += int(np.sum(prod > 0))
        disc += int(np.sum(prod < 0))

    res = stats.kendalltau(t, x)
    tau = float(res[0])
    pval = float(res[1])
    return RichResult(
        payload={
            "tau": tau,
            "S": int(conc - disc),
            "p_value": pval,
            "n_concordant": conc,
            "n_discordant": disc,
            "n": n,
            "method": "Kendall (1938) rank-correlation trend test",
        }
    )


def cheatsheet():
    return "glmsT: Kendall rank-correlation trend test"


# compact alias per ledger/NAMING.md
lineartrend = linear_trend
