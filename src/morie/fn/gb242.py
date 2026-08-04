# morie.fn -- function file (rootcoder007/morie)
"""PDF of the r-th order statistic from a continuous cdf -- Theorem 2.4.2."""

import math

from ._richresult import RichResult

__all__ = ['ostatpdf', 'gibbons_order_pdf']


def ostatpdf(x, r, n, cdf, pdf):
    """Density of X_(r) for a continuous parent.

    Theorem 2.4.2 (book p. 37), eq. (2.4.2):

    .. math::
        f_{X_{(r)}}(x) = \\frac{n!}{(r-1)!(n-r)!}
            [F_X(x)]^{r-1} [1 - F_X(x)]^{n-r} f_X(x).

    Parameters
    ----------
    x : float
        Point at which the density is evaluated.
    r, n : int
        Order-statistic index and sample size, 1 <= r <= n.
    cdf, pdf : callable or float
        F_X and f_X, either callables or their values at ``x``.

    Returns
    -------
    RichResult
        keys ``pdf``, ``coef`` (the multinomial constant), ``fx``,
        ``dx``, ``r``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Theorem 2.4.2, eq. (2.4.2), p. 37.
    """
    r = int(r)
    n = int(n)
    if not 1 <= r <= n:
        raise ValueError("need 1 <= r <= n.")
    fx = float(cdf(x)) if callable(cdf) else float(cdf)
    dx = float(pdf(x)) if callable(pdf) else float(pdf)
    coef = math.factorial(n) / (math.factorial(r - 1) * math.factorial(n - r))
    val = coef * fx ** (r - 1) * (1.0 - fx) ** (n - r) * dx
    return RichResult(
        payload={
            "pdf": float(val),
            "coef": float(coef),
            "fx": fx,
            "dx": dx,
            "r": r,
            "n": n,
            "method": "f_(r)(x) = n!/((r-1)!(n-r)!) F^(r-1) (1-F)^(n-r) f",
        }
    )


gibbons_order_pdf = ostatpdf
