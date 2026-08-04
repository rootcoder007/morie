"""Gaussian-approximation domain check: x must be well inside sqrt(n).

Morin (2016), Probability: For the Enthusiastic Beginner, eq (7.5).
"""

import math

from ._richresult import RichResult

__all__ = ["gaussdom"]


def gaussdom(x, n):
    """Gaussian-approximation domain check: x must be well inside sqrt(n).

    ``well_inside`` is the decision |x|/sqrt(n) < 0.1.

    Parameters
    ----------
    x : float
        Deviation.
    n : int
        Number of trials, >= 1.

    Returns
    -------
    RichResult
        Keys: ratio, well_inside.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (7.5).
    """
    n_i = int(n)
    if n_i < 1:
        raise ValueError("n must be >= 1")
    ratio = abs(float(x)) / math.sqrt(n_i)
    payload = {"ratio": ratio, "well_inside": ratio < 0.1}
    lines = [("x/sqrt(n)", ratio)]
    return RichResult(
        title="Gaussian-approximation domain check: x must be well inside sqrt(n).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "gaussdom: Gaussian-approximation domain check x << sqrt(n). Morin (2016) eq (7.5)."
