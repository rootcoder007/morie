# morie.fn -- function file (rootcoder007/morie)
"""Distribution of type-1 run lengths alone."""

from math import comb, factorial

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gibbons_type1_run_lengths"]


def gibbons_type1_run_lengths(lengths1, n1=None, n2=None):
    r"""Theorem 3.3.2: the marginal probability of a specific multiset
    of type-1 run lengths, irrespective of how type 2 arranges.

    .. math:: f = \\frac{\\dfrac{r_1!}{\\prod_j e_{1j}!}
              \\binom{n_2+1}{r_1}}{\\binom{n_1+n_2}{n_1}},

    the gaps factor C(n2+1, r1) counting where the r_1 type-1 runs
    can sit among the type-2 elements.

    Parameters
    ----------
    lengths1 : array-like of int
        Type-1 run lengths; must sum to n1.
    n1 : int, optional
        Type-1 count; inferred if omitted.
    n2 : int
        Type-2 count (required -- it cannot be inferred).

    Returns
    -------
    RichResult
        keys: ``pmf``, ``r1``, ``n1``, ``n2``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Theorem 3.3.2.
    """
    L1 = np.asarray(lengths1, dtype=int).ravel()
    if np.any(L1 < 1):
        raise ValueError("run lengths must be positive integers.")
    r1 = L1.size
    if r1 < 1:
        raise ValueError("need at least one run.")
    s1 = int(L1.sum())
    if n1 is not None and int(n1) != s1:
        raise ValueError(f"lengths1 sum to {s1}, not n1 = {n1}.")
    n1 = s1
    if n2 is None:
        raise ValueError("n2 is required; it cannot be inferred from lengths1.")
    n2 = int(n2)
    if n2 < 0 or r1 > n2 + 1:
        raise ValueError("r1 runs cannot fit into n2 + 1 gaps.")

    _, counts = np.unique(L1, return_counts=True)
    perms = factorial(r1)
    for cnt in counts:
        perms //= factorial(int(cnt))
    pmf = perms * comb(n2 + 1, r1) / comb(n1 + n2, n1)
    return RichResult(
        payload={"pmf": float(pmf), "r1": r1, "n1": n1, "n2": n2,
                 "method": "Type-1 run-lengths pmf (Gibbons Theorem 3.3.2)"}
    )


def cheatsheet():
    return "gb332: multiset-perms(L1) C(n2+1, r1)/C(n, n1); gaps factor"
