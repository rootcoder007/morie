# morie.fn -- function file (rootcoder007/morie)
"""Efficacy of the Mann-Whitney / Wilcoxon rank-sum test -- eq. (13.3.10)."""

import math

from ._richresult import RichResult

__all__ = ['effwrs', 'gibbons_wrs_efficacy']


def effwrs(m, n, integral):
    """e(U_{m,n}) for the two-sample location problem.

    Book p. 494, eq. (13.3.10):

    .. math:: e(U_{m,n}) = \\frac{12mn
        \\left[\\int_{-\\infty}^{\\infty} f_X^2(x)dx\\right]^2}
        {m+n+1}.

    Because the Mann-Whitney and rank-sum statistics differ only by a
    constant, this is the rank-sum efficacy too.  Unlike the
    one-sample case the book notes that F_X need not be symmetric
    here, so the expression may be evaluated for skewed parents such
    as the exponential.

    Parameters
    ----------
    m, n : int
        The two sample sizes.
    integral : float
        The integral of the squared parent density.

    Returns
    -------
    RichResult
        keys ``efficacy``, ``integral``, ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eq. (13.3.10), p. 494.
    """
    m = int(m)
    n = int(n)
    ii = float(integral)
    if m < 1 or n < 1:
        raise ValueError("m and n must be at least 1.")
    e = 12.0 * m * n * ii * ii / (m + n + 1.0)
    return RichResult(
        payload={
            "efficacy": float(e),
            "integral": ii,
            "m": m,
            "n": n,
            "method": "Mann-Whitney / rank-sum efficacy, eq. (13.3.10)",
        }
    )


gibbons_wrs_efficacy = effwrs
