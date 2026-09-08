# morie.fn -- function file (rootcoder007/morie)
"""Asymptotic distribution of Page's L -- eq. (12.3.2)."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['pageasymp', 'gibbons_page_asymp']


def pageasymp(ell, k, n, correct=True):
    """Normal approximation to Page's L with continuity correction.

    Book p. 449, eq. (12.3.2):

    .. math:: Z = \\frac{12(L - 0.5) - 3kn(n+1)^2}
        {n(n+1)\\sqrt{k(n-1)}},

    approximately standard normal for large k and n, with the
    rejection region in the right tail.  Equivalently
    E[L] = kn(n+1)^2/4 and Var[L] = kn^2(n+1)^2(n-1)/144, both
    returned so the standardisation can be checked directly.

    Parameters
    ----------
    ell : float
        Observed L.
    k : int
        Number of blocks.
    n : int
        Number of treatments, n >= 2.
    correct : bool, optional
        Apply the 0.5 continuity correction (default True).

    Returns
    -------
    RichResult
        keys ``z``, ``p_value``, ``mean``, ``var``, ``statistic``,
        ``k``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eq. (12.3.2), p. 449 (Page, 1963).
    """
    ell = float(ell)
    k = int(k)
    n = int(n)
    if k < 1:
        raise ValueError("k must be at least 1.")
    if n < 2:
        raise ValueError("n must be at least 2.")
    e = ell - 0.5 if correct else ell
    z = (12.0 * e - 3.0 * k * n * (n + 1.0) ** 2) / (
        n * (n + 1.0) * math.sqrt(k * (n - 1.0))
    )
    mean = k * n * (n + 1.0) ** 2 / 4.0
    var = k * float(n) ** 2 * (n + 1.0) ** 2 * (n - 1.0) / 144.0
    return RichResult(
        payload={
            "z": float(z),
            "p_value": float(1.0 - stats.norm.cdf(z)),
            "mean": float(mean),
            "var": float(var),
            "statistic": ell,
            "k": k,
            "n": n,
            "method": "Page's L normal approximation, eq. (12.3.2)",
        }
    )


gibbons_page_asymp = pageasymp
