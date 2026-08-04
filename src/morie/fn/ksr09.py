# morie.fn -- function file (rootcoder007/morie)
"""Z-estimator for a location parameter."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["zestim", "kosorok_z_estimator"]


def zestim(x, kind="huber", k=1.345, iters=200):
    """Z-estimator: the theta solving the estimating equation Psi_n(theta) = 0.

    A Z-estimator is defined by a ZERO, not by a maximum, which is what
    makes the class large enough to hold the median and the Huber
    estimator alongside the mean.  The root is found by bisection with
    a FIXED iteration count rather than a tolerance, so the two
    language arms land on bit-identical iterates -- a tolerance-based
    loop can stop one step apart and disagree in the last few digits.

    Formula: Psi_n(theta) = n^-1 sum_i psi(x_i - theta) = 0, with
             psi(u) = u                    (mean)
             psi(u) = sign(u)              (median)
             psi(u) = max(-k, min(k, u))   (Huber)

    Parameters
    ----------
    x : array-like
        The sample.
    kind : {"mean", "median", "huber"}
        Which estimating function.
    k : float
        Huber tuning constant, k > 0.
    iters : int
        Bisection steps (fixed budget).

    Returns
    -------
    RichResult
        ``estimate``, ``psi_at_estimate``, ``lower``, ``upper``,
        ``iters``, ``n``.

    References
    ----------
    Kosorok (2008), Introduction to Empirical Processes and
    Semiparametric Inference, Section 2.2.5 and Theorem 10.16, which
    define theta_hat_n as an approximate zero of Psi_n(theta) = P_n
    psi_theta with Psi(theta_0) = 0.  Fetched as the full text of the
    book.  The Huber psi is Huber (1964), Robust estimation of a
    location parameter, Annals of Mathematical Statistics 35(1),
    73-101.
    """
    x = C.vec(x)
    n = len(x)
    if n < 1:
        raise ValueError("the sample must be non-empty")
    kind = str(kind).lower()
    k = float(k)
    if kind == "huber" and k <= 0:
        raise ValueError("the Huber constant k must be positive")

    def psi(u):
        if kind == "mean":
            return u
        if kind == "median":
            return 1.0 if u > 0 else (-1.0 if u < 0 else 0.0)
        if kind == "huber":
            return max(-k, min(k, u))
        raise ValueError("kind must be 'mean', 'median' or 'huber'")

    def Psi(th):
        return sum(psi(v - th) for v in x) / n

    lo = min(x)
    hi = max(x)
    if lo == hi:
        return RichResult(payload={
            "estimate": lo, "psi_at_estimate": 0.0, "lower": lo,
            "upper": hi, "iters": 0.0, "n": float(n),
            "method": "Z-estimator, Kosorok Section 2.2.5"})
    a = lo
    b = hi
    it = int(iters)
    for _ in range(it):
        m = 0.5 * (a + b)
        if Psi(a) * Psi(m) <= 0:
            b = m
        else:
            a = m
    th = 0.5 * (a + b)
    return RichResult(payload={
        "estimate": th, "psi_at_estimate": Psi(th), "lower": a, "upper": b,
        "iters": float(it), "n": float(n),
        "method": "Z-estimator, Kosorok Section 2.2.5"})


kosorok_z_estimator = zestim


def cheatsheet():
    return "ksr09: solve n^-1 sum psi(x_i - theta) = 0 by fixed-budget bisection"
