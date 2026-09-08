# morie.fn -- function file (rootcoder007/morie)
"""Null distribution of the total number of runs."""

from math import comb

from ._richresult import RichResult

__all__ = ["gibbons_total_runs_dist"]


def gibbons_total_runs_dist(r, n1, n2):
    r"""Theorem 3.2.2: the pmf of R = R_1 + R_2 under randomness.

    Even r = 2k (the two counts equal):

    .. math:: f_R(r) = \frac{2\binom{n_1-1}{k-1}\binom{n_2-1}{k-1}}
              {\binom{n_1+n_2}{n_1}}

    Odd r = 2k + 1 (counts differ by one, two ways):

    .. math:: f_R(r) = \frac{\binom{n_1-1}{k}\binom{n_2-1}{k-1}
              + \binom{n_1-1}{k-1}\binom{n_2-1}{k}}
              {\binom{n_1+n_2}{n_1}}

    Parameters
    ----------
    r : int
        Total runs, 2 <= r <= n1 + n2.
    n1, n2 : int
        Counts of each type.

    Returns
    -------
    RichResult
        keys: ``pmf``, ``r``, ``n1``, ``n2``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Theorem 3.2.2.
    """
    r, n1, n2 = int(r), int(n1), int(n2)
    if n1 < 1 or n2 < 1:
        raise ValueError("n1 and n2 must be at least 1.")
    if not 2 <= r <= n1 + n2:
        raise ValueError(f"r must lie in 2..{n1 + n2}, got {r}.")
    denom = comb(n1 + n2, n1)
    if r % 2 == 0:
        k = r // 2
        num = 2 * comb(n1 - 1, k - 1) * comb(n2 - 1, k - 1)
    else:
        k = (r - 1) // 2
        num = comb(n1 - 1, k) * comb(n2 - 1, k - 1) + comb(n1 - 1, k - 1) * comb(
            n2 - 1, k
        )
    return RichResult(
        payload={"pmf": float(num / denom), "r": r, "n1": n1, "n2": n2,
                 "method": "Total-runs pmf (Gibbons Theorem 3.2.2)"}
    )


def cheatsheet():
    return "gb322: even r doubles the symmetric term; odd r sums the two offsets"
