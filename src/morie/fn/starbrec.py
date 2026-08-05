"""Stars-and-bars recursion: N_U_n = sum_{j=0}^{n} (N-1)_U_j.

Morin (2016), Probability: For the Enthusiastic Beginner,
eqs (1.51)-(1.52), (1.54)-(1.55).
"""

from . import _morin
from ._richresult import RichResult

__all__ = ["starbrec"]


def starbrec(n, N):
    """Group unordered sets by how many times one chosen letter appears.

    If letter A appears j times, the remaining n-j picks come from the
    other N-1 letters, so N_U_n = sum_{j=0}^{n} (N-1)_U_(n-j).  That is
    eq (1.54) in general, eq (1.51) for n=4, N=3 and eq (1.52) for
    general n with N=3.  Eq (1.55) rewrites each term with eq (1.16);
    the sum is returned alongside the closed form.

    Parameters
    ----------
    n : int
        Number of picks.
    N : int
        Number of distinct types, N >= 2 (the recursion peels one off).

    Returns
    -------
    RichResult
        Keys: n_picks, n_types, recursion_sum, closed_form, n_terms, forms_agree.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs. (1.51)-(1.52), (1.54)-(1.55).
    """
    n = _morin._check_nonneg_int(n, "n")
    N = _morin._check_nonneg_int(N, "N")
    if N < 2:
        raise ValueError("the recursion needs N >= 2")
    total = 0
    for j in range(n + 1):
        total += _morin.stars_and_bars(j, N - 1)
    closed = _morin.stars_and_bars(n, N)
    if total != closed:
        raise AssertionError("recursion and closed form disagree")
    payload = {"n_picks": float(n), "n_types": float(N),
               "recursion_sum": float(total), "closed_form": float(closed),
               "n_terms": float(n + 1), "forms_agree": 1.0}
    return RichResult(
        title="Stars-and-bars recursion N_U_n = sum_j (N-1)_U_j.",
        summary_lines=[("n", n), ("N", N), ("sum", total)],
        payload=payload,
    )


def cheatsheet():
    return "starbrec: N_U_n = sum_{j=0}^{n} (N-1)_U_j, the peel-off-one-letter recursion. Morin (2016) eq (1.54)."
