# morie.fn -- function file (rootcoder007/morie)
"""Empirical process and its limiting covariance."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["empproc", "kosorok_empirical_process"]


def empproc(x, t, F):
    """Empirical process G_n(t) = sqrt(n)(F_n(t) - F(t)) at given points.

    The scaling by sqrt(n) is the whole point: F_n - F goes to zero, so
    the interesting object is the one that does not.  The returned
    covariance is that of the LIMIT (a Brownian bridge composed with
    F), not of G_n, and it is singular at the ends because
    F(t)(1 - F(t)) vanishes there -- which is exactly why goodness-of-
    fit tests are insensitive in the tails.

    Formula: F_n(t) = n^-1 sum_i 1{x_i <= t};
             G_n(t) = sqrt(n) [F_n(t) - F(t)];
             cov[G(s), G(t)] = F(s ^ t) - F(s) F(t)

    Parameters
    ----------
    x : array-like
        The sample.
    t : array-like
        Points at which the process is evaluated.
    F : array-like
        The true cdf at those points, non-decreasing and in [0, 1].

    Returns
    -------
    RichResult
        ``Fn``, ``Gn``, ``cov`` (k x k), ``sup_abs``, ``n``, ``k``.

    References
    ----------
    Kosorok (2008), Introduction to Empirical Processes and
    Semiparametric Inference, Section 2.1: G_n(t) = sqrt(n)[F_n(t) -
    F(t)] converges to a mean zero Gaussian process with
    cov[G(s), G(t)] = F(s ^ t) - F(s)F(t), equation (2.5).  Fetched as
    the full text of the book.
    """
    x = C.vec(x)
    t = C.vec(t)
    F = C.vec(F)
    n = len(x)
    k = len(t)
    if n < 1:
        raise ValueError("the sample must be non-empty")
    if len(F) != k:
        raise ValueError("t and F must have the same length")
    if any(v < 0.0 or v > 1.0 for v in F):
        raise ValueError("F must lie in [0, 1]")
    if any(t[i] > t[i + 1] for i in range(k - 1)):
        raise ValueError("t must be non-decreasing")
    Fn = [sum(1 for v in x if v <= t[j]) / n for j in range(k)]
    Gn = [math.sqrt(n) * (Fn[j] - F[j]) for j in range(k)]
    cov = [[min(F[i], F[j]) - F[i] * F[j] for j in range(k)]
           for i in range(k)]
    return RichResult(payload={
        "Fn": Fn, "Gn": Gn, "cov": cov,
        "sup_abs": max(abs(v) for v in Gn), "n": n, "k": k,
        "method": "Empirical process, Kosorok Section 2.1"})


kosorok_empirical_process = empproc


def cheatsheet():
    return "ksr01: G_n(t) = sqrt(n)(F_n - F); cov = F(s^t) - F(s)F(t)"
