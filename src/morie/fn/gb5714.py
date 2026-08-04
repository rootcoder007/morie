# morie.fn -- function file (rootcoder007/morie)
"""Sample size for the signed-rank test -- Gibbons eq. (5.7.15)."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['wsrn', 'gibbons_wsrt_sampsize']


def wsrn(p2, alpha=0.05, beta=0.05, twosided=False):
    """Noether sample size for the Wilcoxon signed-rank test.

    Book p. 206, eq. (5.7.15):

    .. math:: N = \\frac{(z_\\alpha + z_\\beta)^2}{3(p_2 - 0.5)^2},

    where p2 = P(X_i + X_j > 2 M0) under H1.  The book's worked values
    are N = 1151 for p2 = 0.556 and N = 21 for p2 = 0.921, both at
    alpha = 0.05 and power 0.95.  Two-sided tests replace alpha by
    alpha/2.

    Parameters
    ----------
    p2 : float
        P(X_i + X_j > 2 M0) under the alternative, p2 != 0.5.
    alpha : float, optional
        Size (default 0.05).
    beta : float, optional
        Type II error (default 0.05, i.e. power 0.95).
    twosided : bool, optional
        Use alpha/2 (default False).

    Returns
    -------
    RichResult
        keys ``n``, ``n_raw``, ``z_alpha``, ``z_beta``, ``p2``,
        ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eq. (5.7.15), p. 206.
    """
    p2 = float(p2)
    alpha = float(alpha)
    beta = float(beta)
    if p2 == 0.5:
        raise ValueError("p2 must differ from 0.5.")
    if not 0.0 < p2 < 1.0:
        raise ValueError("p2 must lie strictly inside (0, 1).")
    a = alpha / 2.0 if twosided else alpha
    za = stats.norm.ppf(1.0 - a)
    zb = stats.norm.ppf(1.0 - beta)
    nraw = (za + zb) ** 2 / (3.0 * (p2 - 0.5) ** 2)
    return RichResult(
        payload={
            "n": int(math.ceil(nraw)),
            "n_raw": float(nraw),
            "z_alpha": float(za),
            "z_beta": float(zb),
            "p2": p2,
            "method": "signed-rank sample size, eq. (5.7.15)",
        }
    )


gibbons_wsrt_sampsize = wsrn
