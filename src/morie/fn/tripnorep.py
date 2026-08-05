"""Ordered triples without repetition: N^3 - (3N^2 - 2N) = N(N-1)(N-2).

Morin (2016), Probability: For the Enthusiastic Beginner, eq (1.30).
"""

from . import _morin
from ._richresult import RichResult

__all__ = ["tripnorep"]


def tripnorep(N):
    """Count ordered triples with no repeats by subtracting the repeats.

    Of the N^3 ordered triples, 3N^2 - 2N have at least one repeated
    entry (3N^2 counts the three ways to pick which pair matches, and
    subtracts back the 2N triple-repeats double counted).  Eq (1.30)
    asserts the difference equals the falling product N(N-1)(N-2) of
    eq (1.6).

    Parameters
    ----------
    N : int
        Number of distinct objects, N >= 0.

    Returns
    -------
    RichResult
        Keys: n_objects, total, with_repeat, no_repeat, falling_product, forms_agree.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq. (1.30).
    """
    N = _morin._check_nonneg_int(N, "N")
    total = N ** 3
    with_repeat = 3 * N ** 2 - 2 * N
    no_repeat = total - with_repeat
    falling = N * (N - 1) * (N - 2)
    if no_repeat != falling:
        raise AssertionError("eq (1.30) identity failed")
    payload = {"n_objects": float(N), "total": float(total),
               "with_repeat": float(with_repeat), "no_repeat": float(no_repeat),
               "falling_product": float(falling), "forms_agree": 1.0}
    return RichResult(
        title="Ordered triples without repetition, N^3 - (3N^2 - 2N).",
        summary_lines=[("N", N), ("no repeat", no_repeat)],
        payload=payload,
    )


def cheatsheet():
    return "tripnorep: N^3 - (3N^2 - 2N) = N(N-1)(N-2) ordered triples without repetition. Morin (2016) eq (1.30)."
