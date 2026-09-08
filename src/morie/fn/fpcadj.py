# morie.fn -- function file (rootcoder007/morie)
"""Finite population correction."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["fpc", "finite_population_corr"]


def fpc(N, n):
    """Finite population correction factor for sampling without replacement.

    The correction people forget: drawing 900 of 1000 units leaves very
    little uncertainty, and the with-replacement variance formula does
    not know that.  The factor multiplies the VARIANCE, so the standard
    error carries its square root -- returned separately because that
    is the one that gets misapplied.

    Formula: f = n/N (sampling fraction);  fpc = (N - n)/N = 1 - f;
             V(ybar) = (1 - f) S^2 / n,  se scales by sqrt(1 - f)

    Parameters
    ----------
    N : float
        Population size (may be math.inf for an infinite population).
    n : int
        Sample size, 1 <= n <= N.

    Returns
    -------
    RichResult
        ``fpc`` (variance multiplier), ``se_factor`` (its square root),
        ``fraction``, ``N``, ``n``.

    References
    ----------
    Cochran (1977), Sampling Techniques, 3rd edition, Chapter 2, where
    V(ybar) = (S^2/n)(N - n)/N for simple random sampling without
    replacement.  Cross-checked against the reference implementation in
    the CRAN package ``samplingbook`` 1.2.4, whose ``Smean`` uses the
    variance ``(N - n)/N * (1/(n(n-1))) sum (y - ybar)^2``.
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1")
    N = float(N)
    if N < n:
        raise ValueError("N must be at least n")
    if math.isinf(N):
        f = 0.0
        k = 1.0
    else:
        f = n / N
        k = (N - n) / N
    return RichResult(payload={
        "fpc": k, "se_factor": math.sqrt(k), "fraction": f, "N": N, "n": n,
        "method": "Finite population correction (1 - n/N)"})


finite_population_corr = fpc


def cheatsheet():
    return "fpcadj: fpc = (N-n)/N multiplies the VARIANCE; se by its sqrt"
