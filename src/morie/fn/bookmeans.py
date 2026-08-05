"""Means of the book's five-point data set: xbar = 4, ybar = 3.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (6.83).
"""

from . import _array_core as np

from . import _morin

from ._richresult import RichResult

__all__ = ["bookmeans"]


def bookmeans(x=None, y=None):
    """Means of the book's five-point data set: xbar = 4, ybar = 3.

    Parameters
    ----------
    x, y : array-like, optional
        Data vectors; default to the book's five worked points.

    Returns
    -------
    RichResult
        Keys: xbar, ybar.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (6.83).
    """
    x_d = [2.0, 3.0, 3.0, 5.0, 7.0] if x is None else x
    y_d = [1.0, 1.0, 3.0, 4.0, 6.0] if y is None else y
    x_a = np.atleast_1d(np.asarray(x_d, dtype=float))
    y_a = np.atleast_1d(np.asarray(y_d, dtype=float))
    if x_a.size == 0 or y_a.size == 0:
        raise ValueError("x and y must be non-empty")
    payload = {"xbar": float(x_a.mean()), "ybar": float(y_a.mean())}
    lines = [("xbar", payload["xbar"]), ("ybar", payload["ybar"])]
    return RichResult(
        title="Means of the book's five-point data set: xbar = 4, ybar = 3.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "bookmeans: Means of the book's five worked points. Morin (2016) eq (6.83)."
