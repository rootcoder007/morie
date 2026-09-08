# morie.fn -- function file (rootcoder007/morie)
"""Asymptotic distribution of the two-sample KS statistic."""

import math

from ._richresult import RichResult

__all__ = ['ks2asymp', 'gibbons_ks2_asymp']


def ks2asymp(d, m, n):
    """Kolmogorov limit for D_{m,n}.

    Section 6.3 (book p. 241): as m, n grow with m/(m+n) bounded away
    from 0 and 1,

    .. math:: P\\left[\\sqrt{\\tfrac{mn}{m+n}}\\,D_{m,n} > k\\right]
        \\to 2\\sum_{j=1}^{\\infty}(-1)^{j-1} e^{-2j^2k^2},

    the same limiting law as the one-sample statistic (Theorem 4.3.3),
    with the effective sample size mn/(m+n).  The series is summed to
    a fixed 100 terms, which is convergent to well past double
    precision for every k > 0.05.

    Parameters
    ----------
    d : float
        Observed D_{m,n}.
    m, n : int
        The two sample sizes.

    Returns
    -------
    RichResult
        keys ``p_value``, ``k`` (the standardised statistic),
        ``neff``, ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 6.3, p. 241; Theorem 4.3.3,
    p. 108.
    """
    d = float(d)
    m = int(m)
    n = int(n)
    if m < 1 or n < 1:
        raise ValueError("m and n must be at least 1.")
    if d < 0.0:
        raise ValueError("d must be non-negative.")
    neff = m * n / float(m + n)
    k = math.sqrt(neff) * d
    if k <= 0.0:
        pv = 1.0
    else:
        s = 0.0
        for j in range(1, 101):
            s += (-1.0) ** (j - 1) * math.exp(-2.0 * j * j * k * k)
        pv = min(1.0, max(0.0, 2.0 * s))
    return RichResult(
        payload={
            "p_value": float(pv),
            "k": float(k),
            "neff": float(neff),
            "m": m,
            "n": n,
            "method": "two-sample KS asymptotic (Kolmogorov limit)",
        }
    )


gibbons_ks2_asymp = ks2asymp
