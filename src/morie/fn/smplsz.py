# morie.fn -- function file (rootcoder007/morie)
"""Sample size for a target margin of error."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["nsamp", "sample_size_calc"]


def nsamp(e, S, N=float("inf"), level=0.95):
    """Sample size for a stated half-width, with the finite-population step.

    Cochran's two-step form is used deliberately: compute the
    first approximation n_0 ignoring the population size, then correct
    it.  The one-step algebra gives the same answer but hides the fact
    that n_0 is what the precision actually costs and N only ever
    reduces it.  Both are returned.

    Formula: n_0 = z^2 S^2 / e^2;  n = n_0 / (1 + n_0/N), rounded up

    Parameters
    ----------
    e : float
        Desired half-width of the confidence interval, e > 0.
    S : float
        Population standard deviation (an advance estimate).
    N : float
        Population size; math.inf for the infinite-population case.
    level : float
        Confidence level, 0 < level < 1.

    Returns
    -------
    RichResult
        ``n``, ``n0``, ``z``, ``e``, ``S``, ``N``.

    References
    ----------
    Cochran (1977), Sampling Techniques, 3rd edition, Chapter 4:
    n_0 = z^2 S^2 / e^2 with the correction n = n_0/(1 + n_0/N).
    Cross-checked against the reference implementation in the CRAN
    package ``samplingbook`` 1.2.4, whose ``sample.size.mean`` computes
    ``S^2 / (e^2/q^2 + S^2/N)`` -- the same quantity rearranged.
    """
    e = float(e)
    S = float(S)
    if e <= 0:
        raise ValueError("the half-width e must be positive")
    if S < 0:
        raise ValueError("S must be non-negative")
    if not 0.0 < float(level) < 1.0:
        raise ValueError("level must lie strictly between 0 and 1")
    z = C.qnorm((1.0 + float(level)) / 2.0)
    n0 = z * z * S * S / (e * e)
    N = float(N)
    n = n0 if math.isinf(N) else n0 / (1.0 + n0 / N)
    return RichResult(payload={
        "n": float(math.ceil(n - 1e-12)), "n0": n0, "z": z, "e": e, "S": S,
        "N": N, "method": "Cochran sample size n0/(1 + n0/N)"})


sample_size_calc = nsamp


def cheatsheet():
    return "smplsz: n0 = z^2 S^2/e^2; n = n0/(1 + n0/N)"
