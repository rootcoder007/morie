# morie.fn -- function file (rootcoder007/morie)
"""Sukhatme scale test for samples with a common median at zero."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['sukhatme', 'gibbons_sukhatme']


def sukhatme(x, y, alternative="two-sided"):
    """T counts pairs with X strictly between 0 and Y, eq. (9.7.1).

    Section 9.7 (book p. 323).  With both medians adjusted to zero,

    .. math:: T = \\#\\{(i,j): y_j < x_i < 0 \\;\\text{or}\\;
        0 < x_i < y_j\\},

    so T counts the X's that fall inside the corresponding Y on the
    same side of the origin.  Under H0, p = P(D_ij = 1) = 1/4, hence
    E[T] = mn/4, and substituting p1 = p2 = 1/12 into eq. (9.7.5)
    gives

    .. math:: Var[T] = \\frac{mn(N+7)}{48},

    the same variance the book uses in eq. (9.8.2) for the interval.
    Small T (X's inside the Y's) indicates the Y sample is the more
    dispersed one, i.e. theta > 1 in the scale model.

    Parameters
    ----------
    x, y : sequence of float
        The two samples, both centred at a common median of zero.
    alternative : str, optional
        ``"two-sided"``, ``"less"`` or ``"greater"``.

    Returns
    -------
    RichResult
        keys ``statistic``, ``mean``, ``var``, ``z``, ``p_value``,
        ``phat`` (T/mn), ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 9.7, eqs. (9.7.1)-(9.7.7),
    pp. 323-327 (Sukhatme, 1957); variance confirmed by eq. (9.8.2),
    p. 328.
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    m = len(xs)
    n = len(ys)
    if m < 1 or n < 1:
        raise ValueError("both samples must be non-empty.")
    t = 0
    for xi in xs:
        for yj in ys:
            if (yj < xi < 0.0) or (0.0 < xi < yj):
                t += 1
    nn = m + n
    mean = m * n / 4.0
    var = m * n * (nn + 7.0) / 48.0
    z = (t - mean) / math.sqrt(var)
    if alternative == "less":
        pv = stats.norm.cdf(z)
    elif alternative == "greater":
        pv = 1.0 - stats.norm.cdf(z)
    elif alternative == "two-sided":
        pv = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    else:
        raise ValueError("alternative must be two-sided, less or greater.")
    return RichResult(
        payload={
            "statistic": int(t),
            "mean": float(mean),
            "var": float(var),
            "z": float(z),
            "p_value": float(min(1.0, pv)),
            "phat": float(t / (m * n)),
            "m": m,
            "n": n,
            "method": "Sukhatme scale test, eq. (9.7.1)",
        }
    )


gibbons_sukhatme = sukhatme
