# morie.fn -- slice k04 (rootcoder007/morie)
"""Kendall (1938) rank-correlation test for monotone trend.

Source: Kendall, M. G. (1938).  A new measure of rank correlation.
*Biometrika* 30, 81-93.  The 1938 paper is paywalled here; the measure
is quoted in its standard published form, which is unambiguous::

    tau = (P - Q) / (P + Q)

with P the number of concordant and Q the number of discordant pairs.
Applied with the first argument being time this is the Mann-Kendall
trend test, whose statistic is S = P - Q.  Mann, H. B. (1945),
*Econometrica* 13, 245-259, and Kendall (1938) give the null variance
with the usual tie correction::

    Var(S) = [ n(n-1)(2n+5) - sum_g u_g (u_g - 1)(2 u_g + 5) ] / 18

summed over groups of tied x values, and the continuity-corrected
normal deviate::

    Z = (S - 1)/sqrt(Var(S))  if S > 0
    Z = 0                     if S = 0
    Z = (S + 1)/sqrt(Var(S))  if S < 0

    p = 2 (1 - Phi(|Z|)).

The variance and deviate are written out here rather than delegated to
``_stats_core.kendalltau`` so that the Python and R arms compute the
same quantity by the same route; ``kendalltau`` switches to an exact
null distribution for small n, which the R arm would not reproduce.

The previous body of this module was a one-sample Kolmogorov-Smirnov
test against a fitted normal, pasted by the stub generator.  Deleted.
"""

from __future__ import annotations

import math

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["linear_trend"]


def _tie_correction(v):
    total = 0.0
    vals = sorted(float(a) for a in v)
    i = 0
    n = len(vals)
    while i < n:
        j = i
        while j + 1 < n and vals[j + 1] == vals[i]:
            j += 1
        u = j - i + 1
        if u > 1:
            total += u * (u - 1.0) * (2.0 * u + 5.0)
        i = j + 1
    return total


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
        keys: ``tau``, ``S`` (Mann-Kendall concordance excess), ``var_S``,
        ``z``, ``p_value``, ``n_concordant``, ``n_discordant``, ``n``,
        ``method``.
    """
    t = np.asarray(t, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    n = int(t.size)
    if x.size != n:
        raise ValueError("t and x must have the same length")
    if n < 3:
        raise ValueError("need n >= 3")

    order = np.argsort(t)
    xs = x[order]

    conc = 0
    disc = 0
    for i in range(n - 1):
        dx = xs[i + 1 :] - xs[i]
        conc += int(np.sum(dx > 0))
        disc += int(np.sum(dx < 0))
    S = conc - disc
    tau = float(conc - disc) / float(conc + disc) if (conc + disc) else float("nan")

    var_s = (n * (n - 1.0) * (2.0 * n + 5.0) - _tie_correction(xs)) / 18.0
    if var_s <= 0.0:
        z = float("nan")
        p = float("nan")
    else:
        if S > 0:
            z = (S - 1.0) / math.sqrt(var_s)
        elif S < 0:
            z = (S + 1.0) / math.sqrt(var_s)
        else:
            z = 0.0
        p = 2.0 * float(stats.norm.sf(abs(z)))
        p = min(1.0, p)
    return RichResult(
        payload={
            "tau": tau,
            "S": int(S),
            "var_S": float(var_s),
            "z": float(z),
            "p_value": float(p),
            "n_concordant": conc,
            "n_discordant": disc,
            "n": n,
            "method": "Kendall (1938) rank-correlation trend test (Mann-Kendall normal approximation)",
        }
    )


def cheatsheet():
    return "glmsT: Kendall rank-correlation trend test"


# compact alias per ledger/NAMING.md
lineartrend = linear_trend
