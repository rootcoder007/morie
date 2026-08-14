# morie.fn -- function file (rootcoder007/morie)
"""Sharp regression discontinuity by local linear regression."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["sharp_rdd"]


def _wls_side(r, y, w):
    """Weighted line fit in closed form, with an HC0 variance.

    Solved from the five weighted sums rather than through a general
    solver: two arms calling two different least-squares routines is the
    easiest way to lose the last few digits, and the 2-by-2 normal
    equations have an exact solution anyway.

    Returns (a, b, var_a, var_b, n) where a is the value at r = 0 and b
    the slope.
    """
    n = len(r)
    S0 = S1 = S2 = Sy = Sry = 0.0
    for i in range(n):
        wi = w[i]
        S0 += wi
        S1 += wi * r[i]
        S2 += wi * r[i] * r[i]
        Sy += wi * y[i]
        Sry += wi * r[i] * y[i]
    det = S0 * S2 - S1 * S1
    if abs(det) < 1e-300:
        raise ValueError("sharp_rdd: a side has no variation in the running variable")
    a = (S2 * Sy - S1 * Sry) / det
    b = (S0 * Sry - S1 * Sy) / det
    m00 = m01 = m11 = 0.0
    for i in range(n):
        e = y[i] - a - b * r[i]
        c = w[i] * w[i] * e * e
        m00 += c
        m01 += c * r[i]
        m11 += c * r[i] * r[i]
    # A^{-1} = (1/det) [[S2, -S1], [-S1, S0]]; row 1 of A^{-1} is (S2, -S1)/det
    u0, u1 = S2 / det, -S1 / det
    v0, v1 = -S1 / det, S0 / det
    var_a = u0 * u0 * m00 + 2.0 * u0 * u1 * m01 + u1 * u1 * m11
    var_b = v0 * v0 * m00 + 2.0 * v0 * v1 * m01 + v1 * v1 * m11
    return a, b, var_a, var_b, n


def _rdd_sides(x, cutoff, bandwidth, who):
    """Centre, window and triangular-weight the running variable."""
    r = [float(v) - float(cutoff) for v in C.vec(x)]
    h = float(bandwidth)
    if h <= 0.0:
        raise ValueError(who + ": bandwidth must be positive")
    right = [i for i in range(len(r)) if 0.0 <= r[i] <= h]
    left = [i for i in range(len(r)) if -h <= r[i] < 0.0]
    if len(right) < 2 or len(left) < 2:
        raise ValueError(who + ": each side of the cutoff needs at least two points inside the bandwidth")
    w = [max(0.0, 1.0 - abs(v) / h) for v in r]
    return r, w, left, right, h


def sharp_rdd(y, x, cutoff=0.0, bandwidth=1.0):
    """Jump in the conditional mean at the cutoff, local linear, both sides.

    Local LINEAR rather than local constant is not a refinement: a
    kernel mean at a boundary point is biased at first order because the
    data lie on one side only, and the linear term is exactly what
    removes that bias.  The triangular kernel is used because it is the
    boundary-optimal one for this estimand.

    Formula: ``tau = lim_{x -> c+} E[Y|X=x] - lim_{x -> c-} E[Y|X=x]``,
    each limit the intercept of a triangular-kernel weighted line fit on
    its own side of ``c``.

    Parameters
    ----------
    y : array-like
        Outcome.
    x : array-like
        Running variable, same length as ``y``.
    cutoff : float, default 0.0
        Threshold.
    bandwidth : float, default 1.0
        Half-window, positive.

    Returns
    -------
    RichResult
        ``estimate`` (tau), ``tau``, ``se``, ``z``, ``mu_right``,
        ``mu_left``, ``slope_right``, ``slope_left``, ``n_right``,
        ``n_left``, ``bandwidth``.

    References
    ----------
    Hahn, J., Todd, P. & Van der Klaauw, W. (2001).  Identification and
    estimation of treatment effects with a regression-discontinuity
    design.  Econometrica 69(1):201-209.  doi:10.1111/1468-0262.00183.
    """
    yv = [float(v) for v in C.vec(y)]
    xv = [float(v) for v in C.vec(x)]
    if len(yv) == 0:
        raise ValueError("sharp_rdd: y is empty")
    if len(xv) != len(yv):
        raise ValueError("sharp_rdd: x must have one entry per observation")
    r, w, left, right, h = _rdd_sides(xv, cutoff, bandwidth, "sharp_rdd")
    aR, bR, vR, sR, nR = _wls_side([r[i] for i in right], [yv[i] for i in right],
                                   [w[i] for i in right])
    aL, bL, vL, sL, nL = _wls_side([r[i] for i in left], [yv[i] for i in left],
                                   [w[i] for i in left])
    tau = aR - aL
    se = math.sqrt(vR + vL)
    return RichResult(payload={
        "estimate": tau, "tau": tau, "se": se,
        "z": tau / se if se > 0.0 else float("nan"),
        "mu_right": aR, "mu_left": aL, "slope_right": bR, "slope_left": bL,
        "n_right": nR, "n_left": nL, "bandwidth": h,
        "method": "Sharp RDD, triangular-kernel local linear"})


def cheatsheet():
    return "rdksrn: Sharp RDD with local linear regression"

# public names resolved by fn/_lazy_map.json
sharprdd = sharp_rdd
