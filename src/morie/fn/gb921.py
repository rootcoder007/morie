# morie.fn -- function file (rootcoder007/morie)
"""Mood test for scale: squared rank deviations from the median rank."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['moodscale', 'gibbons_mood_scale']


def _tagged(xs, ys):
    """Pooled sample tagged 0 for X, 1 for Y, sorted with X first on ties."""
    t = [(v, 0) for v in xs] + [(v, 1) for v in ys]
    t.sort(key=lambda p: (p[0], p[1]))
    return [1.0 if lab == 0 else 0.0 for _, lab in t]


def _lrmoments(a, m, n):
    """Theorem 7.3.2 moments of sum a_i Z_i under H0."""
    nn = m + n
    abar = sum(a) / nn
    ss = sum((v - abar) ** 2 for v in a)
    return m * abar, m * n * ss / (nn * (nn - 1.0))


def moodscale(x, y):
    """Mood's M_N statistic for a difference in dispersion.

    Section 9.2 (book p. 314), eq. (9.2.1):

    .. math:: M_N = \\sum_{i=1}^{N}\\left(i - \\frac{N+1}{2}\\right)^2 Z_i,

    with Z_i = 1 when the i-th smallest pooled value is an X.  The
    weights are largest in the tails, so a large M_N says the X's are
    the more widely dispersed sample.  When N is odd the middle
    observation gets weight 0, which is what makes the weights exactly
    symmetric.  Book moments, eqs. (9.2.2)-(9.2.3):

    .. math:: E[M_N] = \\frac{m(N^2-1)}{12}, \\qquad
        Var[M_N] = \\frac{mn(N+1)(N^2-4)}{180}.

    Parameters
    ----------
    x, y : sequence of float
        The two samples; the X sample carries the statistic.

    Returns
    -------
    RichResult
        keys ``statistic``, ``mean``, ``var``, ``var_general``
        (Theorem 7.3.2 on the realised scores), ``z``, ``p_value``,
        ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 9.2, eqs. (9.2.1)-(9.2.3),
    pp. 314-316 (Mood, 1954).
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    m = len(xs)
    n = len(ys)
    if m < 1 or n < 1:
        raise ValueError("both samples must be non-empty.")
    nn = m + n
    z = _tagged(xs, ys)
    a = [(i - (nn + 1.0) / 2.0) ** 2 for i in range(1, nn + 1)]
    stat = sum(a[i] * z[i] for i in range(nn))
    mean = m * (nn * nn - 1.0) / 12.0
    var = m * n * (nn + 1.0) * (nn * nn - 4.0) / 180.0
    _, vg = _lrmoments(a, m, n)
    zz = (stat - mean) / math.sqrt(var) if var > 0 else float("nan")
    return RichResult(
        payload={
            "statistic": float(stat),
            "mean": float(mean),
            "var": float(var),
            "var_general": float(vg),
            "z": float(zz),
            "p_value": float(2.0 * (1.0 - stats.norm.cdf(abs(zz)))),
            "m": m,
            "n": n,
            "method": "Mood scale test, eqs. (9.2.1)-(9.2.3)",
        }
    )


gibbons_mood_scale = moodscale
