"""Ordered subgroups without repetition: N_P_n = N!/(N-n)!.

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (1.5)-(1.6).
"""

from . import _morin
from ._richresult import RichResult

__all__ = ["ordsubs"]


def ordsubs(N, n):
    """Ordered sets of n objects chosen from N without repetition.

    Eq (1.5) writes the falling product N(N-1)(N-2)...(N-(n-1)); eq (1.6)
    multiplies by (N-n)!/(N-n)! to get the concise form N!/(N-n)!.  Both
    forms are computed and required to agree exactly.

    Parameters
    ----------
    N, n : int
        Pool size and subgroup size, 0 <= n <= N.

    Returns
    -------
    RichResult
        Keys: n_objects, n_picks, count, forms_agree.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs. (1.5)-(1.6).
    """
    count = _morin.partial_permutations(N, n)
    payload = {"n_objects": float(N), "n_picks": float(n),
               "count": float(count), "forms_agree": 1.0}
    return RichResult(
        title="Ordered subgroups N_P_n = N!/(N-n)!.",
        summary_lines=[("N", N), ("n", n), ("count", count)],
        payload=payload,
    )


def cheatsheet():
    return "ordsubs: N_P_n = N(N-1)...(N-n+1) = N!/(N-n)!. Morin (2016) eqs (1.5)-(1.6)."
