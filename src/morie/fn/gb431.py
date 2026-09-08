# morie.fn -- function file (rootcoder007/morie)
"""Distribution-freeness of the Kolmogorov-Smirnov statistics."""

import math

from ._richresult import RichResult

__all__ = ['ksdistfree', 'gibbons_ks_dist_free']


def ksdistfree(x, cdf):
    """D_n, D+_n, D-_n together with the PIT values that make them free.

    Theorem 4.3.1 (book p. 111): under H0 the sample X is mapped by the
    probability integral transform to Z_j = F_0(X_(j)), which is a
    Uniform(0, 1) order statistic whatever F_0 is.  Since

    .. math:: D_n^+ = \\max_j (j/n - Z_j), \\quad
              D_n^- = \\max_j (Z_j - (j-1)/n), \\quad
              D_n = \\max(D_n^+, D_n^-),

    depend on the sample only through the Z_j, their null distributions
    are the same for every continuous F_0 -- that is the theorem.

    Parameters
    ----------
    x : sequence of float
        Sample, n >= 1.
    cdf : callable
        The hypothesised continuous cdf F_0.

    Returns
    -------
    RichResult
        keys ``statistic`` (D_n), ``dplus``, ``dminus``, ``z``
        (the PIT values, sorted), ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Theorem 4.3.1, p. 111.
    """
    xs = sorted(float(v) for v in x)
    n = len(xs)
    if n < 1:
        raise ValueError("x must be non-empty.")
    z = [float(cdf(v)) for v in xs]
    dp = max((i + 1) / n - z[i] for i in range(n))
    dm = max(z[i] - i / n for i in range(n))
    return RichResult(
        payload={
            "statistic": float(max(dp, dm)),
            "dplus": float(dp),
            "dminus": float(dm),
            "z": z,
            "n": n,
            "method": "KS statistics via the PIT (Gibbons Thm 4.3.1)",
        }
    )


gibbons_ks_dist_free = ksdistfree
