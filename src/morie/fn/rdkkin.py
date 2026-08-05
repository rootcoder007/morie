# morie.fn -- function file (rootcoder007/morie)
"""Regression kink design: the ratio of slope changes at the threshold."""

import math

from . import _tail1core as C

from ._richresult import RichResult
from .rdksrn import _rdd_sides, _wls_side

__all__ = ["kink_rdd"]


def kink_rdd(y, x, D=None, cutoff=0.0, bandwidth=1.0):
    """Effect identified by a kink -- a slope change, not a level jump.

    ``morie.fn.rgknd`` is the other implementation of this design in the
    tree.  It is not aliased here because its standard error comes from a
    500-draw bootstrap on Python's native generator stream, which no R
    arm can reproduce; the point estimate is deterministic but the
    reported uncertainty is not.  This version keeps the same estimand
    and gives it an analytic HC0 standard error via the delta method, so
    both language arms agree to the last digits.

    With ``D`` omitted the design is sharp: the assignment slope changes
    by exactly one at the threshold and the estimator is just the change
    in the outcome slope.

    Formula: ``tau = (b_Y+ - b_Y-) / (b_D+ - b_D-)`` where each ``b`` is
    the slope of a triangular-kernel weighted local linear fit on its own
    side of the threshold.

    Parameters
    ----------
    y : array-like
        Outcome.
    x : array-like
        Running variable.
    D : array-like, optional
        Assignment/dose; omit for the sharp kink, where the denominator
        is 1 by construction.
    cutoff : float, default 0.0
        Kink threshold.
    bandwidth : float, default 1.0
        Half-window, positive.

    Returns
    -------
    RichResult
        ``estimate`` (tau), ``tau``, ``se``, ``slope_Y_right``,
        ``slope_Y_left``, ``slope_D_right``, ``slope_D_left``,
        ``first_stage`` (the denominator), ``n_right``, ``n_left``,
        ``bandwidth``.

    References
    ----------
    Card, D., Lee, D. S., Pei, Z. & Weber, A. (2015).  Inference on
    causal effects in a generalized regression kink design.
    Econometrica 83(6):2453-2483.  doi:10.3982/ECTA11224.
    """
    yv = [float(v) for v in C.vec(y)]
    xv = [float(v) for v in C.vec(x)]
    if len(yv) == 0:
        raise ValueError("kink_rdd: y is empty")
    if len(xv) != len(yv):
        raise ValueError("kink_rdd: x must have one entry per observation")
    r, w, left, right, h = _rdd_sides(xv, cutoff, bandwidth, "kink_rdd")
    rr = [r[i] for i in right]
    rl = [r[i] for i in left]
    wr = [w[i] for i in right]
    wl = [w[i] for i in left]
    _, byR, _, sYR, nR = _wls_side(rr, [yv[i] for i in right], wr)
    _, byL, _, sYL, nL = _wls_side(rl, [yv[i] for i in left], wl)
    num = byR - byL
    vn = sYR + sYL
    if D is None:
        bdR, bdL, vd = 1.0, 0.0, 0.0
    else:
        dv = [float(v) for v in C.vec(D)]
        if len(dv) != len(yv):
            raise ValueError("kink_rdd: D must have one entry per observation")
        _, bdR, _, sDR, _ = _wls_side(rr, [dv[i] for i in right], wr)
        _, bdL, _, sDL, _ = _wls_side(rl, [dv[i] for i in left], wl)
        vd = sDR + sDL
    den = bdR - bdL
    if abs(den) < 1e-10:
        raise ValueError("kink_rdd: no kink in assignment; the denominator is zero")
    tau = num / den
    se = math.sqrt(vn / (den * den) + (num * num) * vd / (den ** 4))
    return RichResult(payload={
        "estimate": tau, "tau": tau, "se": se,
        "z": tau / se if se > 0.0 else float("nan"),
        "slope_Y_right": byR, "slope_Y_left": byL,
        "slope_D_right": bdR, "slope_D_left": bdL, "first_stage": den,
        "n_right": nR, "n_left": nL, "bandwidth": h,
        "method": "Regression kink design, ratio of local linear slope changes"})


def cheatsheet():
    return "rdkkin: Regression kink design (RKD)"
