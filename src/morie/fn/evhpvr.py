# morie.fn -- function file (rootcoder007/morie)
"""Heffernan-Tawn conditional extremes model."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["evt_heffernan_tawn"]


def _ht_nll(xv, yv, a, b):
    """Profile negative log-likelihood at (a, b), mu and sigma profiled out."""
    n = len(xv)
    z = [(yv[i] - a * xv[i]) / xv[i] ** b for i in range(n)]
    m = sum(z) / n
    s2 = sum((v - m) ** 2 for v in z) / n
    if s2 <= 0.0:
        return float("inf"), m, 0.0
    return (0.5 * n * math.log(s2) + b * sum(math.log(v) for v in xv),
            m, math.sqrt(s2))


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
    a_lo, a_hi, b_lo, b_hi = -1.0, 1.0, 0.0, 0.999
    best = (float("inf"), 0.0, 0.0)
    for i in range(41):
        a = a_lo + (a_hi - a_lo) * i / 40.0
        for j in range(41):
            b = b_lo + (b_hi - b_lo) * j / 40.0
            f = _ht_nll(xv, yv, a, b)[0]
            if f < best[0]:
                best = (f, a, b)
    a, b = best[1], best[2]
    da = (a_hi - a_lo) / 40.0
    db = (b_hi - b_lo) / 40.0
    gr = 0.5 * (math.sqrt(5.0) - 1.0)
    for _ in range(60):
        lo, hi = a - da, a + da
        for _ in range(40):
            c = hi - gr * (hi - lo)
            d = lo + gr * (hi - lo)
            if _ht_nll(xv, yv, c, b)[0] < _ht_nll(xv, yv, d, b)[0]:
                hi = d
            else:
                lo = c
        a = 0.5 * (lo + hi)
        lo, hi = max(b - db, b_lo), min(b + db, b_hi)
        for _ in range(40):
            c = hi - gr * (hi - lo)
            d = lo + gr * (hi - lo)
            if _ht_nll(xv, yv, a, c)[0] < _ht_nll(xv, yv, a, d)[0]:
                hi = d
            else:
                lo = c
        b = 0.5 * (lo + hi)
        da *= 0.5
        db *= 0.5
    nll, mu, sd = _ht_nll(xv, yv, a, b)
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
