# morie.fn -- function file (rootcoder007/morie)
"""Van der Waerden test using inverse-normal scores."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['vdw', 'gibbons_vdw_test']


def vdw(x, y):
    """Van der Waerden X_1 test with scores Phi^{-1}(i/(N+1)).

    Section 8.3.2 (book p. 301).  The van der Waerden statistic
    replaces the exact expected normal order statistics of the
    Terry-Hoeffding test by the far cheaper approximation

    .. math:: a_i = \\Phi^{-1}\\!\\left(\\frac{i}{N+1}\\right),

    and sums them over the ranks of the X sample.  The two tests have
    the same asymptotic behaviour; the scores here need no quadrature
    at all.  Moments from Theorem 7.3.2: mean m * abar, variance
    mn * sum (a_i - abar)^2 / [N(N-1)].

    Parameters
    ----------
    x, y : sequence of float
        The two samples.

    Returns
    -------
    RichResult
        keys ``statistic``, ``z``, ``p_value``, ``mean``, ``var``,
        ``scores``, ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 8.3.2, p. 301
    (van der Waerden, 1952).
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    m = len(xs)
    n = len(ys)
    if m < 1 or n < 1:
        raise ValueError("both samples must be non-empty.")
    nn = m + n
    tagged = [(v, 0) for v in xs] + [(v, 1) for v in ys]
    tagged.sort(key=lambda p: (p[0], p[1]))
    scores = [stats.norm.ppf(i / (nn + 1.0)) for i in range(1, nn + 1)]
    stat = sum(scores[i] for i in range(nn) if tagged[i][1] == 0)
    abar = sum(scores) / nn
    ss = sum((s - abar) ** 2 for s in scores)
    mean = m * abar
    var = m * n * ss / (nn * (nn - 1.0))
    z = (stat - mean) / math.sqrt(var)
    return RichResult(
        payload={
            "statistic": float(stat),
            "z": float(z),
            "p_value": float(2.0 * (1.0 - stats.norm.cdf(abs(z)))),
            "mean": float(mean),
            "var": float(var),
            "scores": scores,
            "m": m,
            "n": n,
            "method": "van der Waerden inverse-normal scores test",
        }
    )


gibbons_vdw_test = vdw
