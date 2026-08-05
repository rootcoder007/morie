# morie.fn -- function file (rootcoder007/morie)
"""Frequentist confidence interval for an interval-identified parameter."""

from . import _tail1core as C

from ._richresult import RichResult
from .bndvar import bound_variance_term

__all__ = ["bound_frequentist"]


def bound_frequentist(lower, upper, alpha=0.05):
    """Imbens-Manski interval from replicated estimates of the two bounds.

    A thin front end over :func:`~morie.fn.bndvar.bound_variance_term`,
    which is where the Imbens-Manski critical value actually lives; this
    module only reduces two samples of bound estimates to the
    ``(lower_hat, upper_hat, se_lower, se_upper, n)`` that construction
    needs.  Aliasing rather than re-deriving matters here: a second copy
    of the critical value would agree with the first at 1e-9 forever and
    still be a second copy.

    Formula: ``[l_hat - c s_l / sqrt(n), u_hat + c s_u / sqrt(n)]`` with
    ``c`` solving ``Phi(c + sqrt(n) Delta / max(s_l, s_u)) - Phi(-c)
    = 1 - alpha``.

    Parameters
    ----------
    lower, upper : array-like
        Replicated estimates of the lower and upper bound, same length.
    alpha : float, optional
        Miss probability, default 0.05.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``c``, ``z_one_sided``,
        ``z_two_sided``, ``delta``, ``n``.

    References
    ----------
    Imbens, G. W. & Manski, C. F. (2004).  Confidence intervals for
    partially identified parameters.  Econometrica 72(6), 1845-1857,
    equation (6).  doi:10.1111/j.1468-0262.2004.00555.x.
    """
    lo = C.vec(lower)
    hi = C.vec(upper)
    n = len(lo)
    if n < 2:
        raise ValueError("bound_frequentist: need at least two replicates")
    if len(hi) != n:
        raise ValueError("bound_frequentist: lower and upper must have the same length")
    tl = C.mean(lo)
    tu = C.mean(hi)
    if tu < tl:
        raise ValueError("bound_frequentist: mean upper is below mean lower")
    sl = C.sd(lo)
    su = C.sd(hi)
    if sl <= 0.0:
        sl = 1e-12
    if su <= 0.0:
        su = 1e-12
    r = bound_variance_term(tl, tu, sl, su, n, alpha)
    ci = r["ci"]
    return RichResult(payload={
        "lower": float(ci[0]), "upper": float(ci[1]),
        "width": float(ci[1]) - float(ci[0]), "c": float(r["c"]),
        "z_one_sided": float(r["z_one_sided"]),
        "z_two_sided": float(r["z_two_sided"]),
        "delta": float(r["delta"]), "n": n,
        "method": "Frequentist bound with valid coverage"})


def cheatsheet():
    return "bndfre: Imbens-Manski interval from replicated bound estimates"
