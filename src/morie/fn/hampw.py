"""Hampel three-part redescending weight function."""

from __future__ import annotations

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["hampel_three_part"]


def _check(a, b, c, who):
    a = float(a)
    b = float(b)
    c = float(c)
    if not (0.0 < a <= b <= c) or b == c:
        raise ValueError(who + ": the constants must satisfy 0 < a <= b < c")
    return a, b, c


def hampel_three_part(y, a=2.0, b=4.0, c=8.0):
    """The weight w(r) = psi(r) / r of Hampel's three-part psi.

    Hampel, F. R. (1974), "The influence curve and its role in robust
    estimation", *Journal of the American Statistical Association*
    69(346), 383-393, doi:10.1080/01621459.1974.10482962, is the shelf
    citation and the source of the influence-curve argument these
    constants come from.  That paper is closed access with no open copy
    in any repository (Unpaywall reports oa_status "closed"), so the
    exact piecewise form was taken from the reference implementation
    that ships with R: MASS::psi.hampel, from Venables, W. N. and
    Ripley, B. D. (2002), *Modern Applied Statistics with S*, 4th ed.,
    Springer, whose body was printed in this session as

        U <- pmin(abs(u) + 1e-50, c)
        ifelse(U <= a, U, ifelse(U <= b, a, a * (c - U)/(c - b))) / U

    i.e. with r = |u|,

        w(r) = 1                            0 <= r <= a
             = a / r                        a <  r <= b
             = a (c - r) / ((c - b) r)      b <  r <= c
             = 0                            r >  c,

    the weight that multiplies each residual in an IRLS M-estimator.
    MASS's default constants a = 2, b = 4, c = 8 are kept.  The 1e-50
    of the R source is a guard against 0/0 at r = 0; this arm returns
    the limit w(0) = 1 directly instead.

    Parameters
    ----------
    y : array-like
        Residuals, usually already scaled by a robust sigma.
    a, b, c : float
        The three bend points, 0 < a <= b < c.

    Returns
    -------
    estimate : the mean weight
    weights  : one weight per residual, each in [0, 1]
    n_zero   : how many residuals were rejected outright (r > c)
    """
    r = core.vec(y)
    if len(r) == 0:
        raise ValueError("hampel_three_part: y is empty")
    a, b, c = _check(a, b, c, "hampel_three_part")
    w = []
    nz = 0
    tot = 0.0
    for e in r:
        u = e if e >= 0.0 else -e
        if u <= a:
            wi = 1.0
        elif u <= b:
            wi = a / u
        elif u <= c:
            wi = a * (c - u) / ((c - b) * u)
        else:
            wi = 0.0
            nz += 1
        w.append(wi)
        tot += wi
    return RichResult(payload={
        "estimate": tot / len(r),
        "weights": w,
        "n_zero": nz,
        "n": len(r),
        "a": a,
        "b": b,
        "c": c,
        "method": "Hampel three-part redescending weight",
    })


def cheatsheet():
    return "hampw: Hampel three-part redescending weight"
