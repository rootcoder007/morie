# morie.fn -- function file (rootcoder007/morie)
"""Kolmogorov-Smirnov versus Cramer-von Mises on the same sample."""

import math

from ._richresult import RichResult

__all__ = ['kscvmcmp', 'gibbons_ks_cvm_comparison']


def kscvmcmp(x, cdf):
    """Both statistics side by side, with the deviation profile.

    Section 4.9 (book p. 146) contrasts the two: the KS statistic reads
    only the single largest vertical gap between S_n and F_0, while the
    Cramer-von Mises statistic integrates the squared gap over the
    whole line, so it responds to many small deviations that the
    supremum ignores.  This returns both statistics, the location of
    the KS supremum, and the share of W^2 contributed by the point
    where D_n is attained -- the quantitative form of that contrast.

    Parameters
    ----------
    x : sequence of float
        Sample, n >= 2.
    cdf : callable
        The hypothesised continuous cdf F_0.

    Returns
    -------
    RichResult
        keys ``d`` (KS), ``w2`` (CvM), ``argmax`` (index of the KS
        supremum), ``share`` (that point's share of the W^2 sum),
        ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 4.9, p. 146.
    """
    xs = sorted(float(v) for v in x)
    n = len(xs)
    if n < 2:
        raise ValueError("need at least 2 observations.")
    z = [float(cdf(v)) for v in xs]
    devs = []
    for i in range(n):
        devs.append(max((i + 1) / n - z[i], z[i] - i / n))
    d = max(devs)
    arg = devs.index(d)
    terms = [
        (z[j] - (2.0 * (j + 1) - 1.0) / (2.0 * n)) ** 2 for j in range(n)
    ]
    w2 = 1.0 / (12.0 * n) + sum(terms)
    return RichResult(
        payload={
            "d": float(d),
            "w2": float(w2),
            "argmax": int(arg),
            "share": float(terms[arg] / w2),
            "n": n,
            "method": "KS (supremum) vs Cramer-von Mises (integrated)",
        }
    )


gibbons_ks_cvm_comparison = kscvmcmp
