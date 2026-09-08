# morie.fn -- function file (rootcoder007/morie)
"""Power of the median test via the precedence-statistic integral."""

import math

from ._richresult import RichResult

__all__ = ['medtestpow', 'gibbons_median_test_power']


def _simpson(f, nodes=2001):
    """Composite Simpson on (0, 1) with a fixed odd node count."""
    if nodes % 2 == 0:
        nodes += 1
    h = 1.0 / (nodes - 1)
    total = 0.0
    for i in range(nodes):
        u = i * h
        w = 1.0 if i in (0, nodes - 1) else (4.0 if i % 2 else 2.0)
        total += w * f(u)
    return total * h / 3.0


def medtestpow(m, n, r, wcrit, g, nodes=2001):
    """Power of a one-sided precedence (median) test, eqs. (6.4.9)-(6.4.10).

    Book p. 254.  With W_r the number of Y's preceding X_(r), the
    alternative distribution is

    .. math:: P(W_r = i) = \\frac{\\binom{n}{i}}{B(r, m-r+1)}
        \\int_0^1 g(u)^i [1-g(u)]^{n-i} u^{r-1}(1-u)^{m-r}\\,du,
        \\qquad g = F_Y \\circ F_X^{-1},

    and the power of the rejection region W_r < w_alpha is the sum of
    those probabilities for i < w_alpha.  Under H0, g(u) = u.  The
    integral is evaluated by composite Simpson on a fixed grid, so the
    result is reproducible to the last bit in any language.

    Parameters
    ----------
    m, n : int
        Sample sizes.
    r : int
        Index of the X order statistic defining the precedence test.
    wcrit : int
        Rejection region is W_r < wcrit.
    g : callable
        u -> F_Y(F_X^{-1}(u)) on [0, 1]; pass ``lambda u: u`` for H0.
    nodes : int, optional
        Simpson nodes (default 2001, forced odd).

    Returns
    -------
    RichResult
        keys ``power``, ``pmf`` (P(W_r = i), i = 0..n), ``m``, ``n``,
        ``r``, ``wcrit``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eqs. (6.4.9)-(6.4.10), p. 254.
    """
    m = int(m)
    n = int(n)
    r = int(r)
    wcrit = int(wcrit)
    if m < 1 or n < 1:
        raise ValueError("m and n must be at least 1.")
    if not 1 <= r <= m:
        raise ValueError("need 1 <= r <= m.")
    beta = (
        math.gamma(r) * math.gamma(m - r + 1) / math.gamma(m + 1)
    )
    pmf = []
    for i in range(n + 1):
        def integrand(u, i=i):
            gu = float(g(u))
            gu = min(1.0, max(0.0, gu))
            return (
                gu**i
                * (1.0 - gu) ** (n - i)
                * u ** (r - 1)
                * (1.0 - u) ** (m - r)
            )

        pmf.append(math.comb(n, i) * _simpson(integrand, nodes) / beta)
    power = sum(pmf[i] for i in range(min(wcrit, n + 1)))
    return RichResult(
        payload={
            "power": float(power),
            "pmf": pmf,
            "m": m,
            "n": n,
            "r": r,
            "wcrit": wcrit,
            "method": "precedence/median-test power, eqs. (6.4.9)-(6.4.10)",
        }
    )


gibbons_median_test_power = medtestpow
