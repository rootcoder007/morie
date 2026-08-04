# morie.fn -- function file (rootcoder007/morie)
"""Exact null distribution of the two-sided Smirnov statistic."""

import math

from ._richresult import RichResult

__all__ = ['smirnov2', 'gibbons_smirnov_2sided']


def smirnov2(d, m, n):
    """P(D_{m,n} >= d) exactly, by lattice-path counting.

    Section 6.3 (book p. 239).  Under H0 every one of the C(m+n, m)
    arrangements of the pooled sample is equally likely, and each
    corresponds to a monotone lattice path from (0,0) to (m,n).  The
    two-sided statistic exceeds d exactly on the paths that touch the
    boundary |i/m - j/n| >= d, so the exact tail is one minus the
    fraction of paths that stay inside the band.

    Parameters
    ----------
    d : float
        Threshold, 0 < d <= 1.
    m, n : int
        The two sample sizes.

    Returns
    -------
    RichResult
        keys ``sf`` (P(D >= d)), ``cdf``, ``npaths``, ``inside``,
        ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 6.3, p. 239; tabulated as
    Table I, p. 571.
    """
    from .gb631 import _ks2count

    d = float(d)
    m = int(m)
    n = int(n)
    if m < 1 or n < 1:
        raise ValueError("m and n must be at least 1.")
    if not 0.0 < d <= 1.0:
        raise ValueError("d must lie in (0, 1].")
    sf = _ks2count(m, n, d, False)
    total = math.comb(m + n, m)
    return RichResult(
        payload={
            "sf": float(sf),
            "cdf": float(1.0 - sf),
            "npaths": float(total),
            "inside": float((1.0 - sf) * total),
            "m": m,
            "n": n,
            "method": "exact two-sided Smirnov distribution",
        }
    )


gibbons_smirnov_2sided = smirnov2
