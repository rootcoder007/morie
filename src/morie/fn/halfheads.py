"""Stirling approximation to P(n Heads in 2n fair flips): 1/sqrt(pi n).

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (2.65)-(2.66).
"""

from . import _morin
from ._richresult import RichResult

__all__ = ["halfheads"]


def halfheads(n):
    """P(exactly n Heads in 2n fair flips) and its Stirling approximation.

    Exact value C(2n, n)/4^n (eq 2.65) against 1/sqrt(pi n) (eq 2.66).

    Parameters
    ----------
    n : int
        Half the number of flips, >= 1.

    Returns
    -------
    RichResult
        Keys: n, approx, exact, relative_error.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs. (2.65)-(2.66).
    """
    approx = _morin.stirling_half_heads(n)
    exact = _morin.exact_half_heads(n)
    payload = {
        "n": int(n),
        "approx": approx,
        "exact": exact,
        "relative_error": abs(approx - exact) / exact,
    }
    lines = [("1/sqrt(pi n)", approx), ("exact", exact)]
    return RichResult(
        title="Stirling approximation: P(n Heads in 2n flips) ~ 1/sqrt(pi n).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "halfheads: P(n Heads in 2n flips) = C(2n,n)/4^n ~ 1/sqrt(pi n). Morin (2016) eqs (2.65)-(2.66)."
