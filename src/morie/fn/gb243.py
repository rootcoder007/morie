# morie.fn -- function file (rootcoder007/morie)
"""Uniform order statistic U_(r) is Beta(r, n-r+1) -- Theorem 2.4.3."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['ostatbeta', 'gibbons_order_beta']


def ostatbeta(u, r, n):
    """Density, cdf and moments of U_(r) for a Uniform(0, 1) parent.

    Theorem 2.4.3 (book p. 38): U_(r) ~ Beta(r, n - r + 1), so

    .. math::
        f_{U_{(r)}}(u) = \\frac{n!}{(r-1)!(n-r)!} u^{r-1}(1-u)^{n-r},
        \\qquad E[U_{(r)}] = \\frac{r}{n+1}.

    Parameters
    ----------
    u : float
        Point in [0, 1].
    r, n : int
        Order-statistic index and sample size.

    Returns
    -------
    RichResult
        keys ``pdf``, ``cdf``, ``mean``, ``var``, ``alpha``, ``beta``,
        ``r``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Theorem 2.4.3, p. 38.
    """
    r = int(r)
    n = int(n)
    if not 1 <= r <= n:
        raise ValueError("need 1 <= r <= n.")
    u = float(u)
    if not 0.0 <= u <= 1.0:
        raise ValueError("u must lie in [0, 1].")
    a = float(r)
    b = float(n - r + 1)
    coef = math.factorial(n) / (math.factorial(r - 1) * math.factorial(n - r))
    dens = coef * u ** (r - 1) * (1.0 - u) ** (n - r)
    return RichResult(
        payload={
            "pdf": float(dens),
            "cdf": float(stats.beta.cdf(u, a, b)),
            "mean": a / (a + b),
            "var": a * b / ((a + b) ** 2 * (a + b + 1.0)),
            "alpha": a,
            "beta": b,
            "r": r,
            "n": n,
            "method": "U_(r) ~ Beta(r, n-r+1) (Gibbons Thm 2.4.3)",
        }
    )


gibbons_order_beta = ostatbeta
