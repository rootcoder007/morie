# morie.fn -- function file (rootcoder007/morie)
"""Cramer-von Mises goodness-of-fit statistic W^2."""

import math

from ._richresult import RichResult

__all__ = ['cvmw2', 'gibbons_cramer_von_mises']


def cvmw2(x, cdf):
    """Cramer-von Mises W^2 for a fully specified continuous F_0.

    Problem 4.14 (book p. 150).  Weighting the squared EDF deviation by
    the null density and integrating gives the computing form

    .. math:: W^2 = \\frac{1}{12n} + \\sum_{j=1}^{n}
        \\left[Z_j - \\frac{2j-1}{2n}\\right]^2,
        \\qquad Z_j = F_0(X_{(j)}),

    which uses every deviation rather than only the supremum, and is
    distribution-free by the same PIT argument as the KS statistic.

    Parameters
    ----------
    x : sequence of float
        Sample, n >= 1.
    cdf : callable
        The hypothesised continuous cdf F_0.

    Returns
    -------
    RichResult
        keys ``statistic`` (W^2), ``nw2`` (n W^2), ``z``, ``n``,
        ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Problem 4.14, p. 150.
    """
    xs = sorted(float(v) for v in x)
    n = len(xs)
    if n < 1:
        raise ValueError("x must be non-empty.")
    z = [float(cdf(v)) for v in xs]
    w2 = 1.0 / (12.0 * n) + sum(
        (z[j] - (2.0 * (j + 1) - 1.0) / (2.0 * n)) ** 2 for j in range(n)
    )
    return RichResult(
        payload={
            "statistic": float(w2),
            "nw2": float(n * w2),
            "z": z,
            "n": n,
            "method": "Cramer-von Mises W^2 (Gibbons Problem 4.14)",
        }
    )


gibbons_cramer_von_mises = cvmw2
