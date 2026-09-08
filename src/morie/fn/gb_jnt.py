# morie.fn -- function file (rootcoder007/morie)
"""Joint density of two order statistics X_(r), X_(s), r < s."""

import math

from ._richresult import RichResult

__all__ = ['ostatjoint', 'gibbons_joint_order']


def ostatjoint(x, y, r, s, n, cdf, pdf):
    """Joint pdf of X_(r) and X_(s) at (x, y) with x < y and r < s.

    Section 2.5 (book p. 39):

    .. math::
        f_{r,s}(x,y) = \\frac{n!}{(r-1)!(s-r-1)!(n-s)!}
            F(x)^{r-1}[F(y)-F(x)]^{s-r-1}[1-F(y)]^{n-s} f(x) f(y).

    Parameters
    ----------
    x, y : float
        Arguments, x < y (the density is 0 otherwise).
    r, s : int
        Indices, 1 <= r < s <= n.
    n : int
        Sample size.
    cdf, pdf : callable
        Parent F_X and f_X.

    Returns
    -------
    RichResult
        keys ``pdf``, ``coef``, ``fx``, ``fy``, ``r``, ``s``, ``n``,
        ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 2.5, p. 39.
    """
    r = int(r)
    s = int(s)
    n = int(n)
    if not 1 <= r < s <= n:
        raise ValueError("need 1 <= r < s <= n.")
    x = float(x)
    y = float(y)
    coef = math.factorial(n) / (
        math.factorial(r - 1) * math.factorial(s - r - 1) * math.factorial(n - s)
    )
    if x >= y:
        return RichResult(
            payload={
                "pdf": 0.0, "coef": float(coef), "fx": float("nan"),
                "fy": float("nan"), "r": r, "s": s, "n": n,
                "method": "joint pdf of X_(r), X_(s) -- support requires x < y",
            }
        )
    fx = float(cdf(x)) if callable(cdf) else float(cdf)
    fy = float(cdf(y)) if callable(cdf) else float(cdf)
    dx = float(pdf(x)) if callable(pdf) else float(pdf)
    dy = float(pdf(y)) if callable(pdf) else float(pdf)
    val = (
        coef
        * fx ** (r - 1)
        * (fy - fx) ** (s - r - 1)
        * (1.0 - fy) ** (n - s)
        * dx
        * dy
    )
    return RichResult(
        payload={
            "pdf": float(val),
            "coef": float(coef),
            "fx": fx,
            "fy": fy,
            "r": r,
            "s": s,
            "n": n,
            "method": "f_(r,s)(x,y), Gibbons Sec. 2.5",
        }
    )


gibbons_joint_order = ostatjoint
