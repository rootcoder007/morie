"""Hockey-stick identity: C(n, k) = sum_{j=k-1}^{n-1} C(j, k-1).

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (1.29), (1.56).
"""

from . import _morin
from ._richresult import RichResult

__all__ = ["hockstick"]


def hockstick(n, k):
    """Sum a diagonal of Pascal's triangle down to a single coefficient.

    Eq (1.29), copied as eq (1.56) inside the stars-and-bars induction,
    states C(n,k) = C(n-1,k-1) + C(n-2,k-1) + ... + C(k-1,k-1).  The
    explicit sum and the closed form are both returned.

    Parameters
    ----------
    n, k : int
        Requires 1 <= k <= n.

    Returns
    -------
    RichResult
        Keys: n, k, stick_sum, closed_form, n_terms, forms_agree.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs. (1.29), (1.56).
    """
    total, closed = _morin.hockey_stick(n, k)
    if total != closed:
        raise AssertionError("hockey-stick sum and closed form disagree")
    payload = {"n": float(n), "k": float(k), "stick_sum": float(total),
               "closed_form": float(closed),
               "n_terms": float(int(n) - int(k) + 1), "forms_agree": 1.0}
    return RichResult(
        title="Hockey-stick identity.",
        summary_lines=[("sum", total), ("C(n,k)", closed)],
        payload=payload,
    )


def cheatsheet():
    return "hockstick: C(n,k) = sum_{j=k-1}^{n-1} C(j,k-1). Morin (2016) eqs (1.29), (1.56)."
