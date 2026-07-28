# morie.fn -- function file (rootcoder007/morie)
"""Marginal null distribution of one run count."""

from math import comb

from ._richresult import RichResult

__all__ = ["gibbons_marginal_r1"]


def gibbons_marginal_r1(r1, n1, n2):
    r"""Corollary to Theorem 3.2.1: the marginal pmf of R_1 is

    .. math:: f_{R_1}(r_1) = \frac{\binom{n_1-1}{r_1-1}
              \binom{n_2+1}{r_1}}{\binom{n_1+n_2}{n_1}},

    obtained by summing the joint pmf over the (at most three)
    feasible values of r_2. The C(n2+1, r1) factor counts the ways to
    seat r_1 type-1 runs into the n_2 + 1 gaps around the type-2
    elements.

    Parameters
    ----------
    r1 : int
        Number of type-1 runs.
    n1, n2 : int
        Counts of each type.

    Returns
    -------
    RichResult
        keys: ``pmf``, ``r1``, ``n1``, ``n2``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Corollary 3.2.1.
    """
    r1, n1, n2 = int(r1), int(n1), int(n2)
    if n1 < 1 or n2 < 1:
        raise ValueError("n1 and n2 must be at least 1.")
    if not 1 <= r1 <= n1:
        raise ValueError(f"r1 must lie in 1..{n1}, got {r1}.")
    pmf = comb(n1 - 1, r1 - 1) * comb(n2 + 1, r1) / comb(n1 + n2, n1)
    return RichResult(
        payload={"pmf": float(pmf), "r1": r1, "n1": n1, "n2": n2,
                 "method": "Marginal runs pmf (Gibbons Corollary 3.2.1)"}
    )


def cheatsheet():
    return "gb321c: C(n1-1,r1-1)C(n2+1,r1)/C(n,n1); gaps argument"
