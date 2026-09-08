# morie.fn -- function file (rootcoder007/morie)
"""Combined MTS + MTR Manski-Pepper bounds on E[Y(d)]."""

from . import _tail1core as C
from .bdmnsl import mtsbound
from .bdmnto import mtrbound

from ._richresult import RichResult

__all__ = ["mts_mtr_combined", "mtsmtrcombined"]


def mts_mtr_combined(y, D, y_min, y_max, d=None):
    """Intersection of the MTS and the MTR bound on ``E[Y(d)]``.

    Monotone treatment selection and monotone treatment response are
    separate assumptions, each of which alone identifies an interval
    containing ``E[Y(d)]``.  Maintaining both means the truth lies in
    both intervals, so the joint bound is their intersection:

        lower = max(lower_MTS, lower_MTR)
        upper = min(upper_MTS, upper_MTR)

    Because both component intervals are valid under their own
    assumption, the intersection is non-empty whenever the two
    assumptions are jointly consistent with the data; an empty result is
    therefore evidence against the pair, and is flagged rather than
    silently returned as a negative width.

    The two component bounds are not recomputed here: MTS is
    ``bdmnsl.mtsbound`` and MTR is ``bdmnto.mtrbound``, each the single
    implementation of its own bound in this package.

    Parameters
    ----------
    y : array-like
        Observed outcomes.
    D : array-like
        Observed treatment levels on an ordered scale.
    y_min, y_max : float
        A priori outcome support.
    d : float, optional
        Level whose mean counterfactual is bounded.  Default: the
        largest observed level.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``mts_lower``, ``mts_upper``,
        ``mtr_lower``, ``mtr_upper``, ``empty``, ``n``, ``d``.

    References
    ----------
    Manski, C. F. and Pepper, J. V. (2000), "Monotone instrumental
    variables: with an application to the returns to schooling",
    Econometrica 68(4), 997-1010.  Standard published form; see
    ``bdmnsl`` for the note on the article's availability.
    """
    yv = C.vec(y)
    z = C.vec(D)
    if len(yv) == 0:
        raise ValueError("y is empty")
    if len(z) != len(yv):
        raise ValueError("y and D must have the same length")
    lo, hi = float(y_min), float(y_max)
    if lo > hi:
        raise ValueError("y_min must not exceed y_max")
    lev = float(max(z)) if d is None else float(d)
    s = mtsbound(yv, z, lev, lo, hi)
    r = mtrbound(yv, z, lev, lo, hi)
    lb = max(s["lower"], r["lower"])
    ub = min(s["upper"], r["upper"])
    return RichResult(payload={
        "lower": lb, "upper": ub, "width": ub - lb,
        "mts_lower": s["lower"], "mts_upper": s["upper"],
        "mtr_lower": r["lower"], "mtr_upper": r["upper"],
        "empty": 1.0 if lb > ub else 0.0, "n": len(yv), "d": lev,
        "method": "Combined MTS+MTR bounds (Manski-Pepper 2000)"})


mtsmtrcombined = mts_mtr_combined


def cheatsheet():
    return "mtsmtv: Combined MTS+MTR Manski-Pepper bounds"
