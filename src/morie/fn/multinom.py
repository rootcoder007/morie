"""Multinomial coefficient N!/(n1! n2! ... nk!).

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (1.35), (1.37).
"""

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["multinom"]


def multinom(ns, N=None):
    """Multinomial coefficient N!/(n1! n2! ... nk!).

    When ``N`` exceeds ``sum(ns)`` the leftover items form one extra
    implicit committee (Morin's remark below eq (1.35)).

    Parameters
    ----------
    ns : array-like of int
        Committee sizes.
    N : int, optional
        Total items; defaults to ``sum(ns)``.

    Returns
    -------
    RichResult
        Keys: ns, N, coefficient, assignments (an alias of coefficient,
        kept for the book-coordinate callers of eq (1.35)).

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs. (1.35), (1.37).
    """
    value = _morin.multinomial_coefficient(ns, N)
    ns_i = [int(x) for x in np.atleast_1d(ns)]
    n_tot = int(N) if N is not None else int(np.sum(ns))
    payload = {"ns": ns_i, "N": n_tot, "coefficient": value, "assignments": value}
    lines = [("multinomial coefficient", value)]
    return RichResult(
        title="Multinomial coefficient N!/(n1!...nk!).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "multinom: multinomial coefficient N!/(n1!...nk!). Morin (2016) eqs (1.35), (1.37)."
