# morie.fn -- function file (rootcoder007/morie)
"""Monotone treatment response bounds (alias of :mod:`bdmnto`)."""

from . import _tail1core as C
from .bdmnto import mtrbound

from ._richresult import RichResult

__all__ = ["monotone_treatment_response", "monotonetreatmentresponse"]


def monotone_treatment_response(y, D, direction="increasing", d=None,
                                y_min=None, y_max=None):
    """Manski-Pepper MTR bounds on ``E[Y(d)]``, either direction.

    This module is an ALIAS.  The increasing-response bound is
    implemented once, in ``bdmnto.mtrbound``; this entry point adds the
    direction switch and delegates.  No second copy of the arithmetic
    exists.

    Under monotone treatment response the counterfactual outcome is a
    weakly increasing function of the treatment level, so for a unit
    observed at level ``z_i`` with outcome ``y_i``:

        Y_i(d) >= y_i  when d >= z_i,   Y_i(d) <= y_i  when d <= z_i,

    giving ``L_i = y_i if z_i <= d else ymin``, ``U_i = y_i if z_i >= d
    else ymax`` and ``E[Y(d)] in [mean(L), mean(U)]``.

    ``direction="decreasing"`` asserts the reverse monotonicity.  That is
    the same statement about the level scale run backwards, so it is
    obtained by negating the treatment levels, which is exactly what this
    function does before delegating.

    Parameters
    ----------
    y : array-like
        Observed outcomes.
    D : array-like
        Observed treatment levels on an ordered scale.
    direction : {"increasing", "decreasing"}
        Direction of the assumed response monotonicity.
    d : float, optional
        Level whose mean counterfactual is bounded.  Default: the
        largest observed level.
    y_min, y_max : float, optional
        A priori outcome support; default the observed range.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``nfixed``, ``n``, ``d``.

    References
    ----------
    Manski, C. F. (1997), "Monotone treatment response", Econometrica
    65(6), 1311-1334; Manski, C. F. and Pepper, J. V. (2000),
    Econometrica 68(4), 997-1010.  Standard published form; see
    ``bdmnto`` for the note on the articles' availability.
    """
    yv = C.vec(y)
    z = C.vec(D)
    if len(yv) == 0:
        raise ValueError("y is empty")
    if len(z) != len(yv):
        raise ValueError("y and D must have the same length")
    if direction not in ("increasing", "decreasing"):
        raise ValueError("direction must be 'increasing' or 'decreasing'")
    lo = float(min(yv)) if y_min is None else float(y_min)
    hi = float(max(yv)) if y_max is None else float(y_max)
    lev = float(max(z)) if d is None else float(d)
    if direction == "increasing":
        r = mtrbound(yv, z, lev, lo, hi)
    else:
        r = mtrbound(yv, [-v for v in z], -lev, lo, hi)
    return RichResult(payload={
        "lower": r["lower"], "upper": r["upper"], "width": r["width"],
        "nfixed": r["nfixed"], "n": r["n"], "d": lev,
        "method": "Monotone treatment response bounds (Manski 1997)"})


monotonetreatmentresponse = monotone_treatment_response


def cheatsheet():
    return "mtrbnd: Manski-Pepper monotone treatment response bounds (alias of bdmnto)"
