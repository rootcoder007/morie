# morie.fn -- function file (rootcoder007/morie)
"""Asymptotic normality of X_(r) when r/n -> p -- Theorem 2.10.1."""

import math

from ._richresult import RichResult

__all__ = ['ostatasymp', 'gibbons_asymp_order_normal']


def ostatasymp(p, n, xp, fxp):
    """Large-sample normal law for the sample p-th quantile.

    Theorem 2.10.1 (book p. 60): if r/n -> p with 0 < p < 1 and the
    parent density f is continuous and positive at the population
    quantile x_p = F^{-1}(p), then

    .. math::
        X_{(r)} \\sim AN\\left(x_p,\\;
            \\frac{p(1-p)}{n\\,[f(x_p)]^{2}}\\right).

    Parameters
    ----------
    p : float
        Quantile level, 0 < p < 1.
    n : int
        Sample size.
    xp : float
        Population quantile x_p.
    fxp : float
        Parent density at x_p, strictly positive.

    Returns
    -------
    RichResult
        keys ``mean``, ``var``, ``se``, ``p``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Theorem 2.10.1, p. 60.
    """
    p = float(p)
    n = int(n)
    fxp = float(fxp)
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly inside (0, 1).")
    if n < 1:
        raise ValueError("n must be at least 1.")
    if fxp <= 0.0:
        raise ValueError("fxp must be strictly positive.")
    var = p * (1.0 - p) / (n * fxp * fxp)
    return RichResult(
        payload={
            "mean": float(xp),
            "var": float(var),
            "se": float(math.sqrt(var)),
            "p": p,
            "n": n,
            "method": "X_(r) ~ AN(x_p, p(1-p)/(n f(x_p)^2)) (Thm 2.10.1)",
        }
    )


gibbons_asymp_order_normal = ostatasymp
