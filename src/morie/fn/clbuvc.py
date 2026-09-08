# morie.fn -- function file (rootcoder007/morie)
"""CLUB: contrastive log-ratio upper bound on mutual information."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["club_upper_bound"]


def club_upper_bound(x, y, q=None):
    """
    CLUB upper bound on mutual information

    Formula: I_CLUB = E[log q(y|x)] - E_marg[log q(y|x)]

    The second term averages the SAME conditional density over
    mismatched pairs, so the bound is the average log-ratio between
    matched and mismatched likelihoods.  It is an upper bound on I(X;Y)
    whenever q equals the true conditional.  With a Gaussian q fitted by
    least squares the value is available in closed form for a bivariate
    normal, -0.5 log(1 - rho^2), which is the reference used here.

    Parameters
    ----------
    x : array-like
        n observations of X (scalar).
    y : array-like
        n observations of Y (scalar).
    q : sequence or None
        (a, b, sigma2) of the conditional model y | x ~ N(a + b x,
        sigma2).  None fits it by least squares.

    Returns
    -------
    result : dict
        Keys: estimate (CLUB bound), club, positive, negative,
        a, b, sigma2, rho, mi_gauss, n.

    References
    ----------
    Cheng, Hao, Dai, Liu, Gan & Carin (2020), CLUB: A Contrastive
    Log-ratio Upper Bound of Mutual Information, ICML 119:1779-1788.
    """
    xs = core.vec(x)
    ys = core.vec(y)
    n = len(xs)
    if n < 3:
        raise ValueError("need at least three observations")
    if len(ys) != n:
        raise ValueError("x and y must have the same length")
    if q is None:
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((v - mx) ** 2 for v in xs)
        sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
        if sxx <= 0.0:
            raise ValueError("x has zero variance; the conditional is undefined")
        b = sxy / sxx
        a = my - b * mx
        s2 = sum((ys[i] - a - b * xs[i]) ** 2 for i in range(n)) / n
    else:
        qq = core.vec(q)
        if len(qq) != 3:
            raise ValueError("q must be (a, b, sigma2)")
        a, b, s2 = qq[0], qq[1], qq[2]
    if not (s2 > 0.0):
        raise ValueError("sigma2 must be strictly positive")

    def lp(yi, xi):
        return -0.5 * (math.log(2.0 * math.pi * s2) + (yi - a - b * xi) ** 2 / s2)

    pos = sum(lp(ys[i], xs[i]) for i in range(n)) / n
    neg = 0.0
    for i in range(n):
        for j in range(n):
            neg += lp(ys[j], xs[i])
    neg /= float(n * n)
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum((v - mx) ** 2 for v in xs) / n)
    sy = math.sqrt(sum((v - my) ** 2 for v in ys) / n)
    rho = (sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / n / (sx * sy)
           if sx > 0.0 and sy > 0.0 else 0.0)
    mi = -0.5 * math.log(1.0 - rho * rho) if abs(rho) < 1.0 else float("inf")
    return RichResult(payload={
        "estimate": pos - neg,
        "club": pos - neg,
        "positive": pos,
        "negative": neg,
        "a": a,
        "b": b,
        "sigma2": s2,
        "rho": rho,
        "mi_gauss": mi,
        "n": n,
        "method": "CLUB contrastive log-ratio upper bound on MI",
    })


def cheatsheet():
    return "clbuvc: CLUB upper bound on mutual information"


# compact alias per ledger/NAMING.md
clubupperbound = club_upper_bound
