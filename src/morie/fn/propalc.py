# morie.fn -- function file (rootcoder007/morie)
"""Proportional allocation of a sample across strata."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["propalloc", "proportional_allocation"]


def propalloc(Nh, n):
    """Allocate n units to strata in proportion to stratum size.

    Proportional allocation is the case in which the stratified mean
    collapses to the ordinary sample mean, which is why it is the
    default even though Neyman allocation is more efficient whenever
    the stratum variances differ.

    The exact n_h are rarely integers.  Largest-remainder apportionment
    is used rather than rounding each independently, because rounding
    can miss the target total by several units, and ties are broken on
    the lowest stratum index so the two language arms agree exactly.

    Formula: n_h = n W_h = n N_h / N, apportioned by largest remainder

    Parameters
    ----------
    Nh : array-like
        Population size of each stratum.
    n : int
        Total sample size to allocate.

    Returns
    -------
    RichResult
        ``nh``, ``nh_exact``, ``Wh``, ``fraction``, ``N``, ``n``,
        ``L``.

    References
    ----------
    Cochran (1977), Sampling Techniques, 3rd edition, Section 5.3,
    Corollary 2, "stratification with proportional allocation of the
    n_h", n_h/n = N_h/N.  Chapter 5 read from the scanned original.
    Cross-checked against the reference implementation in the CRAN
    package ``samplingbook`` 1.2.4, whose ``stratasamp(type = "prop")``
    sets ``wh <- Nh/N``.
    """
    Nh = C.vec(Nh)
    n = int(n)
    L = len(Nh)
    if L < 1:
        raise ValueError("at least one stratum is required")
    if any(v <= 0 for v in Nh):
        raise ValueError("stratum sizes must be positive")
    if n < 0:
        raise ValueError("n must be non-negative")
    N = sum(Nh)
    W = [v / N for v in Nh]
    exact = [n * w for w in W]
    base = [int(v) for v in exact]
    rem = n - sum(base)
    order = sorted(range(L), key=lambda i: (-(exact[i] - base[i]), i))
    for i in order[:rem]:
        base[i] += 1
    return RichResult(payload={
        "nh": base, "nh_exact": exact, "Wh": W, "fraction": n / N,
        "N": N, "n": n, "L": L,
        "method": "Proportional allocation (largest remainder)"})


proportional_allocation = propalloc


def cheatsheet():
    return "propalc: n_h = n N_h/N, largest-remainder apportioned"
