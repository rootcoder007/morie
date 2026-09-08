# morie.fn -- function file (rootcoder007/morie)
"""Heffernan-Tawn conditional extremes model."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["evt_heffernan_tawn"]



def _ht_prof(xv, yv, b):
    """Objective at b with a, mu and sigma all profiled out.

    For fixed b the residual variance is a quadratic in a, so the
    minimising a is the ordinary-least-squares slope of u = y/x^b on
    v = x^(1-b) about their means.  Profiling a out ANALYTICALLY is what
    makes the two language arms agree: the likelihood is flat along the
    (a, b) ridge, and a numerical search there decides its steps on the
    last bits, which walked the arms 1e-8 apart.
    """
    n = len(xv)
    u = [0.0] * n
    v = [0.0] * n
    for i in range(n):
        p = xv[i] ** b
        u[i] = yv[i] / p
        v[i] = xv[i] / p
    mu_u = 0.0
    mu_v = 0.0
    for i in range(n):
        mu_u += u[i]
        mu_v += v[i]
    mu_u /= n
    mu_v /= n
    svv = 0.0
    suv = 0.0
    for i in range(n):
        dv = v[i] - mu_v
        svv += dv * dv
        suv += dv * (u[i] - mu_u)
    a = suv / svv if svv > 0.0 else 0.0
    s2 = 0.0
    for i in range(n):
        r = u[i] - a * v[i]
        s2 += r * r
    m = mu_u - a * mu_v
    s2 = s2 / n - m * m
    if s2 <= 0.0:
        return float("inf"), a, m, 0.0
    slx = 0.0
    for i in range(n):
        slx += math.log(xv[i])
    return 0.5 * n * math.log(s2) + b * slx, a, m, math.sqrt(s2)


def evt_heffernan_tawn(X, u):
    """
    Heffernan-Tawn conditional extremes model

    Formula: Y_j | X = x  =  a_j x + x^(b_j) Z_j   for x > u

    Fitted by Gaussian likelihood on the exceedance set, with the
    residual mean and sd profiled out at every (a, b), leaving a
    two-dimensional bounded search over a in [-1, 1] and b in [0, 1).
    A deterministic grid then a golden-section refinement in each
    coordinate keeps both language arms on the same optimum.

    Parameters
    ----------
    X : array-like
        n x 2 matrix.  Column 0 is the conditioning variable, already on
        a standard Laplace or exponential-type scale.
    u : float
        Conditioning threshold applied to column 0.

    Returns
    -------
    result : dict
        Keys: a, b, mu_z, sigma_z, estimate (a), n_exceed, n.

    References
    ----------
    Heffernan & Tawn (2004), JRSS B 66(3):497-546.
    """
    M = core.mat(X)
    n = len(M)
    if n == 0:
        raise ValueError("empty input: X has no rows")
    if len(M[0]) != 2:
        raise ValueError("X must have exactly two columns")
    u = float(u)
    xv = [M[i][0] for i in range(n) if M[i][0] > u]
    yv = [M[i][1] for i in range(n) if M[i][0] > u]
    k = len(xv)
    if k < 3:
        raise ValueError("fewer than three exceedances of u; nothing to fit")
    if any(v <= 0.0 for v in xv):
        raise ValueError("the conditioning variable must be positive above u")
    b_lo, b_hi = 0.0, 0.999

    def dg(b):
        h = 1e-4
        lo = b - h if b - h > b_lo else b_lo
        hi = b + h if b + h < b_hi else b_hi
        return (_ht_prof(xv, yv, hi)[0] - _ht_prof(xv, yv, lo)[0]) / (hi - lo)

    # bracket the stationary point of the profile by a sign change of the
    # derivative on a fixed grid, then bisect: the derivative crosses zero
    # transversally even where the objective itself is flat, so the sign
    # tests are decided far above the noise floor.
    grid = 200
    prev_b = b_lo
    prev = dg(prev_b)
    b = None
    for i in range(1, grid + 1):
        cb = b_lo + (b_hi - b_lo) * i / grid
        cur = dg(cb)
        if prev <= 0.0 <= cur or cur <= 0.0 <= prev:
            lo, hi = prev_b, cb
            flo = prev
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                fm = dg(mid)
                if (flo <= 0.0) == (fm <= 0.0):
                    lo, flo = mid, fm
                else:
                    hi = mid
            b = 0.5 * (lo + hi)
            break
        prev_b, prev = cb, cur
    if b is None:
        best = (float("inf"), b_lo)
        for i in range(grid + 1):
            cb = b_lo + (b_hi - b_lo) * i / grid
            f = _ht_prof(xv, yv, cb)[0]
            if f < best[0]:
                best = (f, cb)
        b = best[1]
    nll, a, mu, sd = _ht_prof(xv, yv, b)
    return RichResult(payload={
        "a": a,
        "b": b,
        "mu_z": mu,
        "sigma_z": sd,
        "estimate": a,
        "nll": nll,
        "n_exceed": k,
        "n": n,
        "method": "Heffernan-Tawn conditional extremes model",
    })


def cheatsheet():
    return "evhpvr: Heffernan-Tawn conditional extremes model"


# compact alias per ledger/NAMING.md
evtheffernantawn = evt_heffernan_tawn
