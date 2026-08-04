# morie.fn -- function file (rootcoder007/morie)
"""Probability-proportional-to-size selection and its estimator."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["ppssamp", "lottery_sampling"]


def ppssamp(z, y, n, seed=1):
    """Draw n units with probability proportional to size, and estimate.

    The "lottery" or cumulative-total method: lay the sizes end to end,
    throw a dart, take the unit it lands in.  Drawing WITH replacement
    is what makes the Hansen-Hurwitz estimator unbiased and its
    variance a plain sum of squares, and it is also why the same unit
    can appear twice -- that is correct, not a bug.

    Selection uses a pinned Lehmer generator so the two language arms
    draw the SAME units; vary ``seed`` for a different sample.
    Indices returned are ONE-BASED, matching the R arm.

    Formula: p_i = z_i / sum_j z_j;  Yhat = (1/n) sum_i y_i / p_i;
             v(Yhat) = sum_i (y_i/p_i - Yhat)^2 / (n (n - 1))

    Parameters
    ----------
    z : array-like
        Size measure of every unit in the population, strictly
        positive.
    y : array-like
        Value of the variable of interest for every unit, same length
        as z.
    n : int
        Number of draws (with replacement), n >= 2.
    seed : int
        Seed for the pinned generator.

    Returns
    -------
    RichResult
        ``index`` (one-based, with repeats), ``p``, ``estimate``,
        ``se``, ``true_total``, ``n``, ``N``.

    References
    ----------
    Cochran (1977), Sampling Techniques, 3rd edition, Chapter 9A,
    which introduces selection with probability proportional to size by
    the cumulative-total method and the Hansen-Hurwitz estimator
    Yhat = (1/n) sum y_i/p_i with variance
    sum (y_i/p_i - Yhat)^2 / (n(n-1)).  Chapter 9A was NOT in the
    scanned excerpt available to this batch, so the standard published
    form (Hansen & Hurwitz 1943, Annals of Mathematical Statistics
    14(4), 333-362) is used.
    """
    z = C.vec(z)
    y = C.vec(y)
    N = len(z)
    if len(y) != N:
        raise ValueError("z and y must have the same length")
    if any(v <= 0 for v in z):
        raise ValueError("sizes must be strictly positive")
    n = int(n)
    if n < 2:
        raise ValueError("a variance needs at least two draws")
    tot = sum(z)
    p = [v / tot for v in z]
    cum = C.cumsum(p)
    g = C.Lcg(seed)
    idx = []
    for _ in range(n):
        u = g.unif()
        j = 0
        while j < N - 1 and u > cum[j]:
            j += 1
        idx.append(j)
    r = [y[i] / p[i] for i in idx]
    est = sum(r) / n
    var = sum((v - est) ** 2 for v in r) / (n * (n - 1))
    return RichResult(payload={
        "index": [i + 1 for i in idx], "p": p, "estimate": est,
        "se": math.sqrt(var), "true_total": sum(y), "n": n, "N": N,
        "method": "PPS with replacement, Hansen-Hurwitz estimator"})


lottery_sampling = ppssamp


def cheatsheet():
    return "lottosi: p_i = z_i/sum z; Yhat = mean(y_i/p_i); v = sum(...)^2/(n(n-1))"
