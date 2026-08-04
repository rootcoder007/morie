# morie.fn -- function file (rootcoder007/morie)
"""Terry-Hoeffding normal-scores test using expected normal order stats."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['normscores', 'gibbons_terry_hoeffding']


def _enos(i, n, lo=-8.0, hi=8.0, nodes=4001):
    """E[Z_(i:n)] for the standard normal, fixed-grid Simpson."""
    if nodes % 2 == 0:
        nodes += 1
    h = (hi - lo) / (nodes - 1)
    coef = math.exp(
        math.lgamma(n + 1.0) - math.lgamma(i) - math.lgamma(n - i + 1.0)
    )
    total = 0.0
    for k in range(nodes):
        z = lo + k * h
        w = 1.0 if k in (0, nodes - 1) else (4.0 if k % 2 else 2.0)
        p = stats.norm.cdf(z)
        total += w * z * p ** (i - 1) * (1.0 - p) ** (n - i) * stats.norm.pdf(z)
    return coef * total * h / 3.0


def normscores(x, y, nodes=4001):
    """Terry-Hoeffding c_1 test: expected normal order statistics as scores.

    Section 8.3.1 (book p. 299).  The scores are the expected values of
    the standard-normal order statistics,

    .. math:: a_i = E[\\xi_{(i:N)}]
        = \\frac{N!}{(i-1)!(N-i)!}\\int z\\,\\Phi(z)^{i-1}
          [1-\\Phi(z)]^{N-i}\\varphi(z)\\,dz,

    and the statistic is their sum over the X ranks.  Under H0 the mean
    is 0 (the scores sum to zero) and, by Theorem 7.3.2,
    Var = mn sum a_i^2 / [N(N-1)].  The expectations are computed by
    composite Simpson on a fixed grid rather than read from a table, so
    they are available for any N.

    Parameters
    ----------
    x, y : sequence of float
        The two samples.
    nodes : int, optional
        Simpson nodes over [-8, 8] (default 4001, forced odd).

    Returns
    -------
    RichResult
        keys ``statistic``, ``z``, ``p_value``, ``mean``, ``var``,
        ``scores``, ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 8.3.1, p. 299
    (Terry, 1952; Hoeffding, 1951); moments Theorem 7.3.2, p. 279.
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
    scores = [_enos(i, nn, nodes=nodes) for i in range(1, nn + 1)]
    stat = sum(scores[i] for i in range(nn) if tagged[i][1] == 0)
    ssq = sum(s * s for s in scores)
    var = m * n * ssq / (nn * (nn - 1.0))
    z = stat / math.sqrt(var)
    return RichResult(
        payload={
            "statistic": float(stat),
            "z": float(z),
            "p_value": float(2.0 * (1.0 - stats.norm.cdf(abs(z)))),
            "mean": 0.0,
            "var": float(var),
            "scores": scores,
            "m": m,
            "n": n,
            "method": "Terry-Hoeffding normal-scores test (Sec. 8.3.1)",
        }
    )


gibbons_terry_hoeffding = normscores
