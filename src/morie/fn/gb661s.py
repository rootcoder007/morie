# morie.fn -- function file (rootcoder007/morie)
"""Noether sample size for the Mann-Whitney test -- eq. (6.6.18)."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['mwun', 'gibbons_mw_sampsize']


def mwun(p, c=0.5, alpha=0.05, beta=0.10, twosided=False):
    """Total sample size N for a Mann-Whitney test of given power.

    Book p. 269, eq. (6.6.18):

    .. math:: N = \\frac{(z_\\alpha + z_\\beta)^2}
        {12\\,c(1-c)(p - 0.5)^2}, \\qquad c = m/N,

    with p = P(Y > X).  Two-sided tests replace alpha by alpha/2.  The
    book's worked example (alpha = 0.05, power 0.90, p = 0.10,
    c = 4/9) gives N = 18.05, hence 19 observations split m = 8,
    n = 11.  At c = 0.5 the formula reduces to the signed-rank
    sample size (5.7.15).

    Parameters
    ----------
    p : float
        P(Y > X) under the alternative, p != 0.5.
    c : float, optional
        Allocation fraction m/N (default 0.5).
    alpha : float, optional
        Size (default 0.05).
    beta : float, optional
        Type II error (default 0.10).
    twosided : bool, optional
        Use alpha/2 (default False).

    Returns
    -------
    RichResult
        keys ``n``, ``n_raw``, ``m``, ``n_y``, ``z_alpha``,
        ``z_beta``, ``c``, ``p``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eq. (6.6.18), p. 269
    (Noether, 1987).
    """
    p = float(p)
    c = float(c)
    alpha = float(alpha)
    beta = float(beta)
    if p == 0.5:
        raise ValueError("p must differ from 0.5.")
    if not 0.0 < c < 1.0:
        raise ValueError("c must lie strictly inside (0, 1).")
    a = alpha / 2.0 if twosided else alpha
    za = stats.norm.ppf(1.0 - a)
    zb = stats.norm.ppf(1.0 - beta)
    nraw = (za + zb) ** 2 / (12.0 * c * (1.0 - c) * (p - 0.5) ** 2)
    ntot = int(math.ceil(nraw))
    mm = int(round(c * ntot))
    return RichResult(
        payload={
            "n": ntot,
            "n_raw": float(nraw),
            "m": mm,
            "n_y": int(ntot - mm),
            "z_alpha": float(za),
            "z_beta": float(zb),
            "c": c,
            "p": p,
            "method": "Mann-Whitney sample size, eq. (6.6.18)",
        }
    )


gibbons_mw_sampsize = mwun
