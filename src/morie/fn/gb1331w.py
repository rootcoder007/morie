# morie.fn -- function file (rootcoder007/morie)
"""Efficacy of the Wilcoxon signed-rank test -- eq. (13.3.4)."""

import math

from ._richresult import RichResult

__all__ = ['effwsr', 'gibbons_wsrt_efficacy']


def effwsr(n, f0, integral):
    """e(T+) for a parent symmetric about the median.

    Book p. 490, eq. (13.3.4):

    .. math:: e(T^+_N) = \\frac{24\\left[\\frac{f(0)}{N-1}
        + I\\right]^2 N(N-1)^2}{(N+1)(2N+1)},
        \\qquad I = \\int_{-\\infty}^{\\infty} f^2(y)\\,dy,

    where f is the density of the symmetric parent centred at 0.  As
    N grows the f(0)/(N-1) term vanishes and the efficacy approaches
    12 N I^2, the familiar limiting form.

    Parameters
    ----------
    n : int
        Sample size, n >= 2.
    f0 : float
        The density at 0.
    integral : float
        I, the integral of the squared density.

    Returns
    -------
    RichResult
        keys ``efficacy``, ``limit`` (12 N I^2), ``integral``,
        ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eq. (13.3.4), p. 490.
    """
    n = int(n)
    f0 = float(f0)
    ii = float(integral)
    if n < 2:
        raise ValueError("n must be at least 2.")
    e = (
        24.0
        * (f0 / (n - 1.0) + ii) ** 2
        * n
        * (n - 1.0) ** 2
        / ((n + 1.0) * (2.0 * n + 1.0))
    )
    return RichResult(
        payload={
            "efficacy": float(e),
            "limit": float(12.0 * n * ii * ii),
            "integral": ii,
            "n": n,
            "method": "signed-rank efficacy, eq. (13.3.4)",
        }
    )


gibbons_wsrt_efficacy = effwsr
