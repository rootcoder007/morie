"""Probability from a continuous density: the integral of rho over [a, b].

Morin (2016), Probability: For the Enthusiastic Beginner, eq (4.2).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["densprob"]


def densprob(grid, density, a, b):
    """Probability from a continuous density: the integral of rho over [a, b].

    Trapezoid integration on a 513-point refinement of [a, b].  The
    supplied density must integrate to within 5 percent of 1 over the
    whole grid, which catches an unnormalised input.

    Parameters
    ----------
    grid : array-like
        Strictly increasing abscissae, length >= 2.
    density : array-like
        Non-negative density values on ``grid``.
    a, b : float
        Interval bounds, with grid[0] <= a <= b <= grid[-1].

    Returns
    -------
    RichResult
        Keys: probability, a, b.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (4.2).
    """
    value = _morin.density_interval_probability(grid, density, a, b)
    payload = {"probability": value, "a": float(a), "b": float(b)}
    lines = [("P(a <= X <= b)", value)]
    return RichResult(
        title="Probability from a continuous density: the integral of rho over [a, b].",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "densprob: P(a <= X <= b) by trapezoid integration of a density. Morin (2016) eq (4.2)."
