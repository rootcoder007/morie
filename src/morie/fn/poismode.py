"""Poisson mode: P(k) is maximal at k = ceil(a) - 1.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (4.89).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["poismode"]


def poismode(a):
    """Poisson mode: P(k) is maximal at k = ceil(a) - 1.

    Both neighbours of the claimed mode are checked, so a wrong
    tie-breaking rule fails loudly rather than silently.

    Parameters
    ----------
    a : float
        Expected count, > 0.

    Returns
    -------
    RichResult
        Keys: mode, p_mode.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (4.89).
    """
    k_star = _morin.poisson_mode(a)
    p_star = _morin.poisson_pmf(k_star, a)
    lo = _morin.poisson_pmf(k_star - 1, a) if k_star >= 1 else 0.0
    hi = _morin.poisson_pmf(k_star + 1, a)
    if lo > p_star + 1e-15 or hi > p_star + 1e-15:
        raise AssertionError("neighbor beats the claimed mode")
    payload = {"mode": k_star, "p_mode": p_star}
    lines = [("mode k", k_star), ("P(mode)", p_star)]
    return RichResult(
        title="Poisson mode: P(k) is maximal at k = ceil(a) - 1.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "poismode: Poisson mode is ceil(a) - 1. Morin (2016) eq (4.89)."
