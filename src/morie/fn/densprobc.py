"""Centered-interval probability P(T - dT/2 <= X <= T + dT/2).

Morin (2016), Probability: For the Enthusiastic Beginner, eq (4.4).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["densprobc"]


def densprobc(grid, density, center, width):
    """Centered-interval probability P(T - dT/2 <= X <= T + dT/2).

    Parameters
    ----------
    grid, density : array-like
        The density on a strictly increasing grid.
    center, width : float
        The interval is [center - width/2, center + width/2].

    Returns
    -------
    RichResult
        Keys: probability, center, width.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (4.4).
    """
    half = float(width) / 2.0
    value = _morin.density_interval_probability(
        grid, density, float(center) - half, float(center) + half)
    payload = {"probability": value, "center": float(center), "width": float(width)}
    lines = [("P(centered interval)", value)]
    return RichResult(
        title="Centered-interval probability P(T - dT/2 <= X <= T + dT/2).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "densprobc: P over an interval given by its centre and width. Morin (2016) eq (4.4)."
