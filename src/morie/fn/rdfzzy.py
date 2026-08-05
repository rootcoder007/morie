# morie.fn -- function file (rootcoder007/morie)
"""Fuzzy regression discontinuity: the Wald ratio at the cutoff."""

import math

from . import _tail1core as C

from ._richresult import RichResult
from .rdksrn import _rdd_sides, _wls_side

__all__ = ["fuzzy_rdd"]


def fuzzy_rdd(y, x, D, cutoff=0.0, bandwidth=1.0):
    """LATE at the cutoff when the rule only shifts the treatment odds.

    The denominator is the first stage, and it is the whole difficulty:
    when the jump in treatment probability is small the ratio is a weak
    instrument and its standard error understates the real uncertainty.
    The first stage is therefore reported alongside, and a denominator
    indistinguishable from zero is an error rather than an infinity.
    With a denominator of exactly one -- everyone above the cutoff
    treated, nobody below -- the estimator collapses to the sharp one,
    asserted as this module's anchor.

    Formula: ``tau_LATE = (lim Y+ - lim Y-) / (lim D+ - lim D-)``, each
    limit a triangular-kernel weighted local linear intercept.  The
    standard error is the delta method applied to that ratio.

    Parameters
    ----------
    y : array-like
        Outcome.
    x : array-like
        Running variable.
    D : array-like
        Treatment received, same length as ``y``.
    cutoff : float, default 0.0
        Threshold.
    bandwidth : float, default 1.0
        Half-window, positive.

    Returns
    -------
    RichResult
        ``estimate`` (LATE), ``tau``, ``se``, ``reduced_form``,
        ``first_stage``, ``se_reduced_form``, ``se_first_stage``,
        ``n_right``, ``n_left``, ``bandwidth``.

    References
    ----------
    Hahn, J., Todd, P. & Van der Klaauw, W. (2001).  Identification and
    estimation of treatment effects with a regression-discontinuity
    design.  Econometrica 69(1):201-209.  doi:10.1111/1468-0262.00183;
    the fuzzy design is their Theorem 3.
    """
    yv = [float(v) for v in C.vec(y)]
    xv = [float(v) for v in C.vec(x)]
    dv = [float(v) for v in C.vec(D)]
    if len(yv) == 0:
        raise ValueError("fuzzy_rdd: y is empty")
    if len(xv) != len(yv) or len(dv) != len(yv):
        raise ValueError("fuzzy_rdd: x and D must have one entry per observation")
    r, w, left, right, h = _rdd_sides(xv, cutoff, bandwidth, "fuzzy_rdd")
    rr = [r[i] for i in right]
    rl = [r[i] for i in left]
    wr = [w[i] for i in right]
    wl = [w[i] for i in left]
    ayR, _, vyR, _, nR = _wls_side(rr, [yv[i] for i in right], wr)
    ayL, _, vyL, _, nL = _wls_side(rl, [yv[i] for i in left], wl)
    adR, _, vdR, _, _ = _wls_side(rr, [dv[i] for i in right], wr)
    adL, _, vdL, _, _ = _wls_side(rl, [dv[i] for i in left], wl)
    num = ayR - ayL
    den = adR - adL
    if abs(den) < 1e-10:
        raise ValueError("fuzzy_rdd: the first stage is indistinguishable from zero")
    tau = num / den
    vn = vyR + vyL
    vd = vdR + vdL
    # delta method for a ratio, treating numerator and denominator as
    # uncorrelated: the covariance term needs the joint influence
    # function, which a per-side fit does not carry.
    se = math.sqrt(vn / (den * den) + (num * num) * vd / (den ** 4))
    return RichResult(payload={
        "estimate": tau, "tau": tau, "se": se,
        "z": tau / se if se > 0.0 else float("nan"),
        "reduced_form": num, "first_stage": den,
        "se_reduced_form": math.sqrt(vn), "se_first_stage": math.sqrt(vd),
        "n_right": nR, "n_left": nL, "bandwidth": h,
        "method": "Fuzzy RDD, Wald ratio of local linear intercepts"})


def cheatsheet():
    return "rdfzzy: Fuzzy RDD (incomplete compliance at the cutoff)"
