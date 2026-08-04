# morie.fn -- function file (rootcoder007/morie)
"""Exact null distribution of the one-sided Smirnov statistic."""

import math

from ._richresult import RichResult

__all__ = ['smirnov1', 'gibbons_smirnov_one_sided']


def smirnov1(d, m, n):
    """P(D+_{m,n} >= d) exactly, plus the one-sided asymptotic form.

    Section 6.3 (book p. 241).  The one-sided statistic
    D+_{m,n} = sup_x [S_m(x) - S_n(x)] tests the stochastic-ordering
    alternative.  Exact tail by lattice-path counting under the
    one-sided boundary i/m - j/n >= d; the limiting form is

    .. math:: P\\left[\\sqrt{\\tfrac{mn}{m+n}}\\,D^+_{m,n} > k\\right]
        \\to e^{-2k^2}.

    Parameters
    ----------
    d : float
        Threshold, 0 < d <= 1.
    m, n : int
        The two sample sizes.

    Returns
    -------
    RichResult
        keys ``sf``, ``cdf``, ``sf_asymp``, ``k``, ``m``, ``n``,
        ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 6.3, p. 241.
    """
    from .gb631 import _ks2count

    d = float(d)
    m = int(m)
    n = int(n)
    if m < 1 or n < 1:
        raise ValueError("m and n must be at least 1.")
    if not 0.0 < d <= 1.0:
        raise ValueError("d must lie in (0, 1].")
    sf = _ks2count(m, n, d, True)
    k = math.sqrt(m * n / float(m + n)) * d
    return RichResult(
        payload={
            "sf": float(sf),
            "cdf": float(1.0 - sf),
            "sf_asymp": float(math.exp(-2.0 * k * k)),
            "k": float(k),
            "m": m,
            "n": n,
            "method": "exact one-sided Smirnov distribution",
        }
    )


gibbons_smirnov_one_sided = smirnov1
