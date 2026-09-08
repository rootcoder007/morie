# morie.fn -- function file (rootcoder007/morie)
"""Grubbs' test for a single outlier.

Source: Grubbs, F. E. (1969), "Procedures for detecting outlying
observations in samples", *Technometrics* 11(1):1-21.  The Technometrics
article is paywalled and was NOT read directly.  The test was taken from
the reference implementation in the R package **outliers**
(``grubbs.test``, ``qgrubbs``, ``pgrubbs``), whose type-10 branch is
reproduced here exactly.

Statistic, from ``grubbs.test``:

    G = |x_out - mean(x)| / sd(x)

where x_out is whichever of max(x), min(x) lies further from the mean
and sd is the unbiased (n - 1) sample standard deviation.

p-value, from ``qgrubbs(..., rev = TRUE)`` verbatim:

    s   <- (G^2 * n * (2 - n)) / (G^2 * n - (n - 1)^2)
    t   <- sqrt(s)
    res <- n * (1 - pt(t, n - 2)),  capped at 1

so p = res.  Equivalently t^2 = (n-2) n G^2 / ((n-1)^2 - n G^2).  This is
the ONE-SIDED form: the tail probability is multiplied by n, not 2n.
The critical value returned inverts the same expression,

    G_crit = ((n-1)/sqrt(n)) * sqrt(t_a^2 / (n - 2 + t_a^2)),
    t_a = qt(alpha/n, n - 2)

which is ``qgrubbs(1 - alpha, n, type = 10)``.  A caller wanting the
two-sided convention printed in most textbooks (alpha/(2n)) should read
``p_value`` as half of it and halve ``alpha`` before passing it.

DEFECT FIXED.  The previous body of this module computed a
Kolmogorov-Smirnov goodness-of-fit statistic -- sorted data, an
empirical CDF, sup|F_n - F| -- and returned it labelled "Grubbs'
single-outlier test".  It never formed G, never used the maximum
deviation, and its p-value came from the KS null distribution.  The
signature carried a ``cdf`` argument, which Grubbs' test has no use for,
as a leftover of that copy.  ``cdf`` is therefore gone; ``alpha`` now has
a default and is used only for the critical value.
"""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["grubbs_test"]


def grubbs_test(x, alpha=0.05, opposite=False):
    """Grubbs' test that the most extreme observation is an outlier.

    Parameters
    ----------
    x : array-like
        Sample, at least three finite values.
    alpha : float
        Level at which the critical value is reported.
    opposite : bool
        Test the end NOT selected by the maximum deviation, matching the
        ``opposite`` argument of ``outliers::grubbs.test``.

    Returns
    -------
    RichResult
        ``statistic`` (G), ``p_value``, ``critical_value``, ``reject``,
        ``outlier`` (its value), ``index``, ``side``, ``mean``, ``sd``,
        ``n``.
    """
    xs = [float(v) for v in x]
    n = len(xs)
    if n < 3:
        raise ValueError("Grubbs' test needs at least three observations")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    m = sum(xs) / n
    ss = 0.0
    for v in xs:
        dv = v - m
        ss += dv * dv
    sd = math.sqrt(ss / (n - 1))
    if sd <= 0.0:
        raise ValueError("x is constant; Grubbs' statistic is undefined")
    imax = 0
    imin = 0
    for i in range(1, n):
        if xs[i] > xs[imax]:
            imax = i
        if xs[i] < xs[imin]:
            imin = i
    hi = xs[imax] - m
    lo = m - xs[imin]
    take_high = hi >= lo
    if opposite:
        take_high = not take_high
    idx = imax if take_high else imin
    side = "max" if take_high else "min"
    g = abs(xs[idx] - m) / sd
    den = g * g * n - (n - 1.0) ** 2
    if den == 0.0:
        p = 0.0
    else:
        s = (g * g * n * (2.0 - n)) / den
        if s < 0.0:
            s = 0.0
        t = math.sqrt(s)
        p = n * (1.0 - float(stats.t.cdf(t, n - 2)))
        if p > 1.0:
            p = 1.0
        if p < 0.0:
            p = 0.0
    ta = float(stats.t.ppf(float(alpha) / n, n - 2))
    ta2 = ta * ta
    crit = ((n - 1.0) / math.sqrt(n)) * math.sqrt(ta2 / (n - 2.0 + ta2))
    return RichResult(payload={
        "statistic": float(g), "p_value": float(p),
        "critical_value": float(crit), "reject": bool(g > crit),
        "outlier": float(xs[idx]), "index": idx, "side": side,
        "mean": float(m), "sd": float(sd), "alpha": float(alpha), "n": n,
        "method": "Grubbs (1969) single-outlier test, outliers::grubbs.test "
                  "type 10; p = n (1 - pt(t, n-2)), one-sided"})


def cheatsheet():
    return "grubbs: Grubbs (1969) single-outlier test"


# compact alias per ledger/NAMING.md
grubbstest = grubbs_test
