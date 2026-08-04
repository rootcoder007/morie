# morie.fn -- function file (rootcoder007/morie)
"""Systematic sampling: the exact design variance."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["sysamp", "systematic_sampling"]


def sysamp(y, k):
    """Every k-th unit: all k systematic samples and the exact variance.

    A systematic sample has only k possible outcomes, so its design
    variance can be written down exactly rather than estimated -- and
    that is the honest way to teach it, because the usual trap is to
    apply the simple-random-sampling variance formula to a systematic
    sample and get an answer that can be wildly wrong in either
    direction.  ``deff`` is the ratio of the two, so the size of that
    error is the output.

    Formula: V(ybar_sy) = (1/k) sum_{i=1}^{k} (ybar_i - Ybar)^2;
             deff = V(ybar_sy) / [(1 - f) S^2 / n]

    Parameters
    ----------
    y : array-like
        The WHOLE population, in the order it would be sampled.
        len(y) must be an exact multiple of k.
    k : int
        Sampling interval; there are k possible systematic samples of
        size n = N/k.

    Returns
    -------
    RichResult
        ``means`` (the k sample means), ``population_mean``,
        ``variance``, ``se``, ``srs_variance``, ``deff``, ``rho``,
        ``N``, ``n``, ``k``.

    References
    ----------
    Cochran (1977), Sampling Techniques, 3rd edition, Chapter 8, which
    defines the k systematic samples and gives the design variance as
    the variance of their means, together with the intraclass
    correlation form V(ybar_sy) = (S^2/n)[1 + (n - 1) rho].  Chapter 8
    was NOT in the scanned excerpt available to this batch, so the
    standard published form is used; rho is reported by inverting that
    identity rather than by a separate formula.
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
    Yb = sum(y) / N
    means = [sum(y[j] for j in range(i, N, k)) / n for i in range(k)]
    V = sum((m - Yb) ** 2 for m in means) / k
    S2 = C.var(y, 1)
    Vsrs = (1.0 - n / N) * S2 / n
    rho = ((V * n / S2) - 1.0) / (n - 1) if S2 > 0 and n > 1 else float("nan")
    return RichResult(payload={
        "means": means, "population_mean": Yb, "variance": V,
        "se": math.sqrt(V), "srs_variance": Vsrs,
        "deff": V / Vsrs if Vsrs > 0 else float("nan"), "rho": rho,
        "N": N, "n": n, "k": k,
        "method": "Systematic sampling, exact design variance"})


systematic_sampling = sysamp


def cheatsheet():
    return "systmp: V(ybar_sy) = (1/k) sum (ybar_i - Ybar)^2, exact"
