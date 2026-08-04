# morie.fn -- function file (rootcoder007/morie)
"""Kolmogorov-Smirnov confidence band for the population cdf."""

import math

from ._richresult import RichResult

__all__ = ['ksband', 'gibbons_ks_conf_band']


def ksband(x, dcrit, at=None):
    """Lower and upper confidence bands L_n(x), U_n(x) for F_X.

    Section 4.4.2 (book p. 121): from P(D_n > D_{n,alpha}) = alpha,

    .. math:: L_n(x) = \\max[S_n(x) - D_{n,\\alpha}, 0], \\qquad
              U_n(x) = \\min[S_n(x) + D_{n,\\alpha}, 1],

    and F_X lies wholly between them with probability 1 - alpha.  The
    book stresses the truncation at 0 and 1, because the raw inequality
    admits numbers outside [0, 1].

    Parameters
    ----------
    x : sequence of float
        Sample, n >= 1.
    dcrit : float
        The critical value D_{n,alpha}.
    at : sequence of float, optional
        Points at which to report the band (defaults to the sorted
        sample).

    Returns
    -------
    RichResult
        keys ``at``, ``edf``, ``lower``, ``upper``, ``width``,
        ``dcrit``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 4.4.2, p. 121.
    """
    xs = sorted(float(v) for v in x)
    n = len(xs)
    if n < 1:
        raise ValueError("x must be non-empty.")
    dcrit = float(dcrit)
    if not 0.0 < dcrit < 1.0:
        raise ValueError("dcrit must lie strictly inside (0, 1).")
    pts = xs if at is None else [float(v) for v in at]
    edf = []
    for v in pts:
        c = 0
        for xi in xs:
            if xi <= v:
                c += 1
            else:
                break
        edf.append(c / n)
    lo = [max(e - dcrit, 0.0) for e in edf]
    hi = [min(e + dcrit, 1.0) for e in edf]
    return RichResult(
        payload={
            "at": pts,
            "edf": edf,
            "lower": lo,
            "upper": hi,
            "width": [hi[i] - lo[i] for i in range(len(pts))],
            "dcrit": dcrit,
            "n": n,
            "method": "KS confidence band, Sec. 4.4.2",
        }
    )


gibbons_ks_conf_band = ksband
