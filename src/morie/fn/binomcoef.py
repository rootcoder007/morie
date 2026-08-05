"""Unordered subgroups: the binomial coefficient N_C_n = N!/(n!(N-n)!).

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (1.7)-(1.8).
"""

from . import _morin
from ._richresult import RichResult

__all__ = ["binomcoef"]


def binomcoef(N, n):
    """Number of unordered sets of n objects chosen from N, no repetition.

    Eq (1.7) divides the ordered count N_P_n by the n! orderings of each
    set; eq (1.8) names the result the binomial coefficient C(N, n).  The
    ordered count and n! * C(N, n) are cross-checked.

    Parameters
    ----------
    N, n : int
        Pool size and subgroup size, 0 <= n <= N.

    Returns
    -------
    RichResult
        Keys: n_objects, n_picks, count, ordered_count, forms_agree.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs. (1.7)-(1.8).
    """
    N = _morin._check_nonneg_int(N, "N")
    n = _morin._check_nonneg_int(n, "n")
    if n > N:
        raise ValueError("n cannot exceed N")
    count = _morin.binom(N, n)
    ordered = _morin.partial_permutations(N, n)
    if _morin.factorial(n) * count != ordered:
        raise AssertionError("n! C(N,n) does not equal N_P_n")
    payload = {"n_objects": float(N), "n_picks": float(n),
               "count": float(count), "ordered_count": float(ordered),
               "forms_agree": 1.0}
    return RichResult(
        title="Unordered subgroups: binomial coefficient C(N, n).",
        summary_lines=[("N", N), ("n", n), ("count", count)],
        payload=payload,
    )


def cheatsheet():
    return "binomcoef: C(N,n) = N!/(n!(N-n)!) = N_P_n / n!. Morin (2016) eqs (1.7)-(1.8)."
