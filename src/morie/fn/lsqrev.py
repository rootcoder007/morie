"""Reverse least squares: x = C y + D, minimizing horizontal residuals.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (6.50).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["lsqrev"]


def lsqrev(x, y):
    """Reverse least squares: x = C y + D, minimizing horizontal residuals.

    Parameters
    ----------
    x, y : array-like
        Equal-length data vectors, n >= 2.

    Returns
    -------
    RichResult
        Keys: C, D, S.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (6.50).
    """
    C, D, S = _morin.least_squares_fit(y, x)
    payload = {"C": C, "D": D, "S": S}
    lines = [("C", C), ("D", D)]
    return RichResult(
        title="Reverse least squares: x = C y + D, minimizing horizontal residuals.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "lsqrev: Reverse least-squares line x = Cy + D. Morin (2016) eq (6.50)."
