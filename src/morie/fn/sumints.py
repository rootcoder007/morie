"""Sum of the first N integers: 1 + 2 + ... + N = N(N+1)/2.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (1.31).
"""

from . import _morin
from ._richresult import RichResult

__all__ = ["sumints"]


def sumints(N):
    """Triangular number N(N+1)/2, with the inductive step checked.

    Eq (1.31) is the induction step: adding (N+1) to N(N+1)/2 gives
    (N+1)(N+2)/2, the same formula with N -> N+1.  The explicit sum, the
    closed form, and the stepped-up closed form are all returned.

    Parameters
    ----------
    N : int
        Upper limit, N >= 0.

    Returns
    -------
    RichResult
        Keys: n, explicit_sum, closed_form, next_closed_form, forms_agree.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq. (1.31).
    """
    N = _morin._check_nonneg_int(N, "N")
    explicit = sum(range(1, N + 1))
    closed = N * (N + 1) // 2
    nxt = (N + 1) * (N + 2) // 2
    if explicit != closed or closed + (N + 1) != nxt:
        raise AssertionError("the induction step failed")
    payload = {"n": float(N), "explicit_sum": float(explicit),
               "closed_form": float(closed), "next_closed_form": float(nxt),
               "forms_agree": 1.0}
    return RichResult(
        title="Sum of the first N integers = N(N+1)/2.",
        summary_lines=[("N", N), ("sum", closed)],
        payload=payload,
    )


def cheatsheet():
    return "sumints: 1+2+...+N = N(N+1)/2, with the induction step to (N+1)(N+2)/2. Morin (2016) eq (1.31)."
