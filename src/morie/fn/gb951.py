# morie.fn -- function file (rootcoder007/morie)
"""Klotz normal-scores test for scale -- eq. (9.5.1)."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['klotzsc', 'gibbons_klotz_scale']


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


def klotzsc(x, y):
    """K_N with squared inverse-normal scores.

    Section 9.5 (book p. 322), eq. (9.5.1):

    .. math:: K_N = \\sum_{i=1}^{N}
        \\left[\\Phi^{-1}\\!\\left(\\frac{i}{N+1}\\right)\\right]^2 Z_i,

    i.e. the van der Waerden location scores squared, so the Klotz test
    stands to the van der Waerden test exactly as Mood's test stands to
    Wilcoxon's.  The larger weights sit at both extremes, so H0 is
    rejected for large K_N against the alternative that the X
    population has the larger spread.  The book's moments are

    .. math:: E[K_N] = \\frac{m}{N}\\sum_i
        \\left[\\Phi^{-1}\\!\\left(\\tfrac{i}{N+1}\\right)\\right]^2,

    with the variance the corresponding Theorem 7.3.2 expression.

    Parameters
    ----------
    x, y : sequence of float
        The two samples.

    Returns
    -------
    RichResult
        keys ``statistic``, ``mean``, ``var``, ``z``, ``p_value``,
        ``scores``, ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 9.5, eq. (9.5.1), p. 322
    (Klotz, 1962).
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    m = len(xs)
    n = len(ys)
    if m < 1 or n < 1:
        raise ValueError("both samples must be non-empty.")
    nn = m + n
    z = _tagged(xs, ys)
    a = [stats.norm.ppf(i / (nn + 1.0)) ** 2 for i in range(1, nn + 1)]
    stat = sum(a[i] * z[i] for i in range(nn))
    mean, var = _lrmoments(a, m, n)
    zz = (stat - mean) / math.sqrt(var) if var > 0 else float("nan")
    return RichResult(
        payload={
            "statistic": float(stat),
            "mean": float(mean),
            "var": float(var),
            "z": float(zz),
            "p_value": float(2.0 * (1.0 - stats.norm.cdf(abs(zz)))),
            "scores": a,
            "m": m,
            "n": n,
            "method": "Klotz normal-scores scale test, eq. (9.5.1)",
        }
    )


gibbons_klotz_scale = klotzsc
