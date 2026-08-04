"""Hampel three-part redescending psi function."""

from __future__ import annotations

from . import _s03core as core

from ._richresult import RichResult
from .hampw import _check

__all__ = ["hampel_redescend"]


def hampel_redescend(r, a=2.0, b=4.0, c=8.0):
    """Hampel's three-part redescending psi.

    Hampel, F. R. (1974), "The influence curve and its role in robust
    estimation", *Journal of the American Statistical Association*
    69(346), 383-393, doi:10.1080/01621459.1974.10482962, is the shelf
    citation; it is closed access with no open copy in any repository
    (Unpaywall reports oa_status "closed"), so the exact piecewise form
    was taken from the reference implementation shipped with R,
    MASS::psi.hampel (Venables and Ripley, *Modern Applied Statistics
    with S*, 4th ed., 2002), whose body was printed in this session.
    MASS returns the weight psi(u)/u; multiplying it back by u gives

        psi(r) = r                              0 <= |r| <= a
               = a sign(r)                      a <  |r| <= b
               = a (c - |r|) sign(r) / (c - b)  b <  |r| <= c
               = 0                              |r| >  c,

    which is linear, then flat, then descending to zero, then zero --
    the shape the module name promises.  Its derivative, which MASS
    returns for deriv = 1, is 1, 0, -a/(c-b), 0 on the same four
    pieces, and is returned here as ``psi_deriv``.

    Rejecting |r| > c outright is what makes this psi redescending
    rather than merely bounded like Huber's.

    Parameters
    ----------
    r : array-like
        Residuals, usually already scaled by a robust sigma.
    a, b, c : float
        The three bend points, 0 < a <= b < c.

    Returns
    -------
    estimate  : the mean of psi, which an M-estimator drives to zero
    psi       : psi evaluated at each residual
    psi_deriv : the derivative on each piece
    n_reject  : how many residuals fell beyond c
    """
    x = core.vec(r)
    if len(x) == 0:
        raise ValueError("hampel_redescend: r is empty")
    a, b, c = _check(a, b, c, "hampel_redescend")
    ps = []
    dp = []
    nrej = 0
    tot = 0.0
    for e in x:
        u = e if e >= 0.0 else -e
        sg = 1.0 if e >= 0.0 else -1.0
        if u <= a:
            p = e
            d = 1.0
        elif u <= b:
            p = a * sg
            d = 0.0
        elif u <= c:
            p = a * (c - u) / (c - b) * sg
            d = -a / (c - b)
        else:
            p = 0.0
            d = 0.0
            nrej += 1
        ps.append(p)
        dp.append(d)
        tot += p
    return RichResult(payload={
        "estimate": tot / len(x),
        "psi": ps,
        "psi_deriv": dp,
        "n_reject": nrej,
        "n": len(x),
        "a": a,
        "b": b,
        "c": c,
        "method": "Hampel three-part redescender",
    })


def cheatsheet():
    return "hampel: Hampel three-part redescender"
