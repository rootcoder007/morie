# morie.fn -- function file (rootcoder007/morie)
"""Joint distribution of all run lengths."""

from math import comb, factorial

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gibbons_run_lengths_dist"]


def gibbons_run_lengths_dist(lengths1, lengths2, n1=None, n2=None):
    r"""Theorem 3.3.1: the probability of observing a specific
    multiset of run lengths for both types.

    With r_1 type-1 runs of given lengths and r_2 type-2 runs
    (|r_1 - r_2| <= 1 by alternation),

    .. math:: f = \\frac{c\; \\dfrac{r_1!}{\\prod_j e_{1j}!}\;
              \\dfrac{r_2!}{\\prod_j e_{2j}!}}{\\binom{n_1+n_2}{n_1}},

    where the e's count repeats among each type's run lengths (equal
    lengths are indistinguishable as a multiset) and c = 2 if
    r_1 = r_2, else 1.

    Parameters
    ----------
    lengths1, lengths2 : array-like of int
        Run lengths of each type; must sum to n1 and n2.
    n1, n2 : int, optional
        Element counts; inferred from the lengths if omitted.

    Returns
    -------
    RichResult
        keys: ``pmf``, ``r1``, ``r2``, ``c``, ``n1``, ``n2``,
        ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Theorem 3.3.1.
    """
    L1 = np.asarray(lengths1, dtype=int).ravel()
    L2 = np.asarray(lengths2, dtype=int).ravel()
    if np.any(L1 < 1) or np.any(L2 < 1):
        raise ValueError("run lengths must be positive integers.")
    r1, r2 = L1.size, L2.size
    if r1 < 1 or r2 < 1:
        raise ValueError("each type needs at least one run.")
    if abs(r1 - r2) > 1:
        raise ValueError("run counts must alternate: |r1 - r2| <= 1.")
    s1, s2 = int(L1.sum()), int(L2.sum())
    if n1 is not None and int(n1) != s1:
        raise ValueError(f"lengths1 sum to {s1}, not n1 = {n1}.")
    if n2 is not None and int(n2) != s2:
        raise ValueError(f"lengths2 sum to {s2}, not n2 = {n2}.")
    n1, n2 = s1, s2

    def multiset_perms(L):
        _, counts = np.unique(L, return_counts=True)
        out = factorial(len(L))
        for cnt in counts:
            out //= factorial(int(cnt))
        return out

    c = 2 if r1 == r2 else 1
    pmf = c * multiset_perms(L1) * multiset_perms(L2) / comb(n1 + n2, n1)
    return RichResult(
        payload={"pmf": float(pmf), "r1": r1, "r2": r2, "c": c,
                 "n1": n1, "n2": n2,
                 "method": "Run-lengths joint pmf (Gibbons Theorem 3.3.1)"}
    )


def cheatsheet():
    return "gb331: c * multiset-perms(L1) * multiset-perms(L2) / C(n, n1)"
