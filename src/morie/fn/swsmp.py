# morie.fn -- function file (rootcoder007/morie)
"""Systematic sampling with a random start."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["sysrs", "systematic_with_random_start"]


def sysrs(y, k, seed=1):
    """Draw one systematic sample with a random start, and estimate from it.

    The random start is what makes the design unbiased; without it the
    "sample" is just a fixed subset.  The start is drawn from a pinned
    Lehmer generator so the two language arms produce the SAME sample
    -- callers who want a genuinely random start should vary ``seed``.

    Indices returned are ONE-BASED, matching the R arm.

    Formula: r ~ Uniform{1..k};  sample = {r, r+k, r+2k, ...};
             ybar_sy = mean of the sample

    Parameters
    ----------
    y : array-like
        The whole population, in sampling order; len(y) an exact
        multiple of k.
    k : int
        Sampling interval.
    seed : int
        Seed for the pinned generator that picks the start.

    Returns
    -------
    RichResult
        ``start`` (one-based), ``index`` (one-based), ``sample``,
        ``estimate``, ``design_se``, ``population_mean``, ``N``, ``n``,
        ``k``.

    References
    ----------
    Cochran (1977), Sampling Techniques, 3rd edition, Chapter 8: a unit
    is chosen at random from the first k, and every k-th unit
    thereafter.  Chapter 8 was NOT in the scanned excerpt available to
    this batch, so the standard published form is used.  The design
    standard error is the exact one computed by the sibling module
    ``systmp``.
    """
    y = C.vec(y)
    N = len(y)
    k = int(k)
    if k < 1:
        raise ValueError("the interval k must be at least 1")
    if N % k != 0:
        raise ValueError("len(y) must be an exact multiple of k")
    n = N // k
    if n < 2:
        raise ValueError("each systematic sample needs at least two units")
    g = C.Lcg(seed)
    r = int(g.unif() * k)
    if r >= k:
        r = k - 1
    idx = list(range(r, N, k))
    smp = [y[i] for i in idx]
    Yb = sum(y) / N
    means = [sum(y[j] for j in range(i, N, k)) / n for i in range(k)]
    V = sum((m - Yb) ** 2 for m in means) / k
    return RichResult(payload={
        "start": r + 1, "index": [i + 1 for i in idx], "sample": smp,
        "estimate": sum(smp) / n, "design_se": math.sqrt(V),
        "population_mean": Yb, "N": N, "n": n, "k": k,
        "method": "Systematic sample with a pinned random start"})


systematic_with_random_start = sysrs


def cheatsheet():
    return "swsmp: r ~ U{1..k}, take r, r+k, ...; pinned generator"
