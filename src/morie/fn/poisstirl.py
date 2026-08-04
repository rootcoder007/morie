"""Poisson pmf with Stirling's approximation applied to k!.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (5.16).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["poisstirl"]


def poisstirl(k, a):
    """Poisson pmf with Stirling's approximation applied to k!.

    Parameters
    ----------
    k : int
        Event count, >= 1 (Stirling needs a positive factorial).
    a : float
        Expected count, > 0.

    Returns
    -------
    RichResult
        Keys: approx, exact, rel_error.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (5.16).
    """
    approx = _morin.poisson_stirling(k, a)
    exact = _morin.poisson_pmf(k, a)
    payload = {
        "approx": approx,
        "exact": exact,
        "rel_error": abs(approx - exact) / max(exact, 1e-300),
    }
    lines = [("Stirling PP(k)", approx), ("exact", exact)]
    return RichResult(
        title="Poisson pmf with Stirling's approximation applied to k!.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "poisstirl: Poisson pmf with Stirling applied to k!. Morin (2016) eq (5.16)."
