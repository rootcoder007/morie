# morie.fn -- function file (rootcoder007/morie)
"""Monotone treatment selection bounds (alias of :mod:`bdmnsl`)."""

from . import _tail1core as C
from .bdmnsl import mtsbound

__all__ = ["mts_bounds", "mtsbounds"]


def mts_bounds(Y, X, monotone=True, d=None, ymin=None, ymax=None):
    """Manski-Pepper monotone-treatment-selection bounds on ``E[Y(d)]``.

    This module is an ALIAS.  The bound itself is implemented once, in
    ``bdmnsl.mtsbound``; this entry point only supplies the argument
    spelling used by the selection-bounds literature (outcome ``Y``,
    selected level ``X``) and the direction switch, then delegates.  No
    second copy of the arithmetic exists.

    Under monotone treatment selection units that select a higher level
    would have had weakly higher outcomes at every counterfactual level,
    so for ``t'' >= t'``, ``E[Y(t) | X = t''] >= E[Y(t) | X = t']``.  The
    observed conditional mean at ``d`` therefore bounds the
    counterfactual mean from above in the groups that selected less and
    from below in those that selected more:

        E[Y(d)] <= P(X<=d) E[Y|X=d] + P(X>d) ymax
        E[Y(d)] >= P(X<d) ymin      + P(X>=d) E[Y|X=d]

    ``monotone=False`` reverses the selection inequality, which is the
    same bound computed on ``-Y`` over the reflected support, with the
    two sides swapped back.

    Parameters
    ----------
    Y : array-like
        Observed outcomes.
    X : array-like
        Observed (selected) treatment levels.
    monotone : bool, default True
        ``True``: higher selection implies weakly higher outcomes.
        ``False``: the reverse.
    d : float, optional
        Level whose mean counterfactual is bounded.  Defaults to the
        largest observed level.
    ymin, ymax : float, optional
        A priori outcome support.  Default to the observed range.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``condmean``, ``pbelow``,
        ``pat``, ``pabove``, ``n``, ``d``.

    References
    ----------
    Manski, C. F. and Pepper, J. V. (2000), "Monotone instrumental
    variables: with an application to the returns to schooling",
    Econometrica 68(4), 997-1010.  Standard published form; see
    ``bdmnsl`` for the note on the article's availability.
    """
    y = C.vec(Y)
    x = C.vec(X)
    if len(y) == 0:
        raise ValueError("Y is empty")
    if len(x) != len(y):
        raise ValueError("Y and X must have the same length")
    lo = float(min(y)) if ymin is None else float(ymin)
    hi = float(max(y)) if ymax is None else float(ymax)
    lev = float(max(x)) if d is None else float(d)
    if monotone:
        return mtsbound(y, x, lev, lo, hi)
    r = mtsbound([-v for v in y], x, lev, -hi, -lo)
    return type(r)(payload={
        "lower": -r["upper"], "upper": -r["lower"], "width": r["width"],
        "condmean": -r["condmean"], "pbelow": r["pbelow"], "pat": r["pat"],
        "pabove": r["pabove"], "n": r["n"], "d": r["d"],
        "method": "Monotone treatment selection bounds (Manski-Pepper 2000)"})


mtsbounds = mts_bounds


def cheatsheet():
    return "mtsbnd: Monotone treatment selection bounds (alias of bdmnsl.mtsbound)"
