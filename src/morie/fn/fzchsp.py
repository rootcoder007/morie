# morie.fn -- function file (rootcoder007/morie)
"""Chung-Smirnov statistic for the kernel distribution function estimator."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["chungsmir", "fauzi_chung_smirnov"]


def chungsmir(x, cdf, h=None, grid=None):
    r"""Chung-Smirnov statistic for the kernel distribution function estimator.

    Sec. 2.1 records the Chung-Smirnov property of the KDFE:

    .. math:: \limsup_{n\to\infty}\sqrt{\frac{2n}{\log\log n}}\,
              \sup_{x\in\mathbb R}|\hat F_h(x) - F_X(x)| = 1
              \quad\text{a.s.}

    A law of the iterated logarithm, not a distributional limit: it pins
    the ALMOST-SURE fluctuation of the uniform error at exactly
    :math:`\sqrt{\log\log n/(2n)}`, with constant 1, neither 1-epsilon nor
    1+epsilon.

    This routine returns the normalised statistic
    :math:`\sqrt{2n/\log\log n}\sup_x|\hat F_h(x)-F(x)|`, whose limsup is
    the quantity the theorem sets to 1. Values persistently above 1 are
    evidence against :math:`F`; a single value above 1 is not, because a
    limsup is attained infinitely often and exceeded infinitely often on
    the way.

    Undefined for :math:`n \le 15`, where :math:`\log\log n \le 1` makes
    the normaliser meaningless; the routine says so instead of returning a
    number.

    Parameters
    ----------
    x : array-like
        Sample.
    cdf : callable
        The hypothesised ``F(t)``.
    h : float, optional
        Bandwidth; defaults to the distribution-function rule
        ``4^(1/3) sigma n^(-1/3)``.
    grid : array-like, optional
        Points at which the supremum is taken; defaults to the sample.

    Returns
    -------
    RichResult
        Keys ``statistic``, ``supdiff``, ``scale``, ``h``, ``n``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Sec. 2.1, the Chung-Smirnov display.
    """
    from . import _stats_core as stats
    from ._fauzi import kdfe_bandwidth

    xv = np.asarray(x, dtype=float).ravel()
    n = xv.size
    if n <= 15:
        raise ValueError(
            "the Chung-Smirnov normaliser needs log log n > 1, i.e. n > 15; "
            f"got n = {n}."
        )
    if h is None:
        h = kdfe_bandwidth(xv)
    h = float(h)
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    g = np.sort(xv) if grid is None else np.atleast_1d(np.asarray(grid, dtype=float))
    khat = np.asarray(
        [float(np.mean(stats.norm.cdf((float(t) - xv) / h))) for t in g], dtype=float
    )
    fv = np.asarray([float(cdf(float(t))) for t in g], dtype=float)
    sup = float(np.max(np.abs(khat - fv)))
    scale = float(np.sqrt(2.0 * n / np.log(np.log(n))))
    return RichResult(
        payload={
            "statistic": float(scale * sup),
            "supdiff": sup,
            "scale": scale,
            "h": h,
            "n": int(n),
            "method": "Chung-Smirnov normalised uniform error of the KDFE",
        }
    )


fauzi_chung_smirnov = chungsmir


def cheatsheet():
    return "fzchsp: Chung-Smirnov LIL: sqrt(2n/log log n) sup|Fhat - F| has limsup exactly 1"


# CANONICAL TEST
# >>> r = chungsmir(list(range(1, 51)), cdf=lambda t: min(max((t - 1) / 49, 0), 1))
# >>> r['statistic'] == r['scale'] * r['supdiff']
# True
