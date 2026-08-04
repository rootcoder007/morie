# morie.fn -- function file (rootcoder007/morie)
"""Monotone treatment response bounds."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["mtrbound", "bound_monot_outcome"]


def mtrbound(y, z, d, ymin, ymax):
    """Bounds on E[Y(d)] when the response is weakly increasing in treatment.

    Under monotone treatment response the counterfactual outcome is a
    weakly increasing function of the treatment level, Y_i(t') >= Y_i(t)
    whenever t' >= t.  For a unit observed at treatment z_i with outcome
    y_i that pins one side of every counterfactual:

        Y_i(d) >= y_i   when d >= z_i,   Y_i(d) <= y_i   when d <= z_i,

    so the observed outcome itself serves as the bound on the side
    monotonicity fixes, and the a priori support endpoint on the other:

        L_i = y_i if z_i <= d else ymin
        U_i = y_i if z_i >= d else ymax
        E[Y(d)] in [ mean(L), mean(U) ].

    Parameters
    ----------
    y : array-like
        Observed outcomes.
    z : array-like
        Observed treatment levels, on the same ordered scale as ``d``.
    d : float
        Treatment level whose mean counterfactual is bounded.
    ymin, ymax : float
        A priori support of the outcome.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``nfixed``, ``n``, ``d``.

    References
    ----------
    Manski, C. F. (1997), "Monotone treatment response", Econometrica
    65(6), 1311-1334, which is the source of the response monotonicity
    and of the bound above; the companion assumptions are Manski and
    Pepper (2000), Econometrica 68(4), 997-1010.  Standard published
    form; neither article was in the local corpus and neither could be
    downloaded (JSTOR returned an access stub), so neither was read.
    """
    y = C.vec(y)
    z = C.vec(z)
    n = len(y)
    if len(z) != n:
        raise ValueError("y and z must have the same length")
    if n == 0:
        raise ValueError("need at least one unit")
    lo, hi = float(ymin), float(ymax)
    if lo > hi:
        raise ValueError("ymin must not exceed ymax")
    if any(v < lo or v > hi for v in y):
        raise ValueError("observed outcomes must lie in [ymin, ymax]")
    d = float(d)
    L = [y[i] if z[i] <= d else lo for i in range(n)]
    U = [y[i] if z[i] >= d else hi for i in range(n)]
    lb = sum(L) / n
    ub = sum(U) / n
    return RichResult(payload={
        "lower": lb, "upper": ub, "width": ub - lb,
        "nfixed": sum(1 for i in range(n) if z[i] == d), "n": n, "d": d,
        "method": "Monotone treatment response bounds (Manski 1997)"})


bound_monot_outcome = mtrbound


def cheatsheet():
    return "bdmnto: Monotone treatment response bounds."
