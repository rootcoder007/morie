# morie.fn -- function file (rootcoder007/morie)
"""Covariance of two uniform order statistics U_(r), U_(s), r <= s."""

import math

from ._richresult import RichResult

__all__ = ['ostatcov', 'gibbons_order_covariance']


def ostatcov(r, s, n):
    """Cov(U_(r), U_(s)) and the induced correlation.

    Section 2.4 (book p. 38): for r <= s,

    .. math::
        Cov[U_{(r)}, U_{(s)}] = \\frac{r(n-s+1)}{(n+1)^2 (n+2)}.

    Parameters
    ----------
    r, s : int
        Indices with 1 <= r <= s <= n.
    n : int
        Sample size.

    Returns
    -------
    RichResult
        keys ``cov``, ``corr``, ``var_r``, ``var_s``, ``r``, ``s``,
        ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 2.4, p. 38.
    """
    r = int(r)
    s = int(s)
    n = int(n)
    if not 1 <= r <= s <= n:
        raise ValueError("need 1 <= r <= s <= n.")
    den = (n + 1.0) ** 2 * (n + 2.0)
    cov = r * (n - s + 1.0) / den
    vr = r * (n - r + 1.0) / den
    vs = s * (n - s + 1.0) / den
    return RichResult(
        payload={
            "cov": float(cov),
            "corr": float(cov / math.sqrt(vr * vs)),
            "var_r": float(vr),
            "var_s": float(vs),
            "r": r,
            "s": s,
            "n": n,
            "method": "Cov(U_(r),U_(s)) = r(n-s+1)/((n+1)^2 (n+2))",
        }
    )


gibbons_order_covariance = ostatcov
