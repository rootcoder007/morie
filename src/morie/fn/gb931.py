# morie.fn -- function file (rootcoder007/morie)
"""Freund-Ansari-Bradley-David-Barton scale test using folded ranks."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['ansbrad', 'gibbons_fab_test']


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


def ansbrad(x, y):
    """A_N with the folded (absolute-deviation) scores, eq. (9.3.1).

    Section 9.3 (book p. 316):

    .. math:: A_N = \\sum_{i=1}^{N}
        \\left|i - \\frac{N+1}{2}\\right| Z_i,

    the Freund-Ansari-Bradley-David-Barton family.  Giving equal weight
    to positive and negative rank deviations makes the scores symmetric
    about the middle of the array, so large A_N again indicates the X's
    are more dispersed.  Moments come from Theorem 7.3.2 applied to the
    realised scores, which is exact for every m, n.

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
    Gibbons & Chakraborti (2011), Sec. 9.3, eq. (9.3.1), p. 316
    (Freund and Ansari, 1957; Ansari and Bradley, 1960; David and
    Barton, 1958).
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    m = len(xs)
    n = len(ys)
    if m < 1 or n < 1:
        raise ValueError("both samples must be non-empty.")
    nn = m + n
    z = _tagged(xs, ys)
    a = [abs(i - (nn + 1.0) / 2.0) for i in range(1, nn + 1)]
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
            "method": "Freund-Ansari-Bradley-David-Barton test, eq. (9.3.1)",
        }
    )


gibbons_fab_test = ansbrad
