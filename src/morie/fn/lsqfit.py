"""Least-squares line y = Ax + B, with the residual sum of squares.

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (6.42)-(6.49), (6.82).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["lsqfit"]


def lsqfit(x=None, y=None):
    """Least-squares line y = Ax + B, with the residual sum of squares.

    A = (<xy> - <x><y>)/(<x^2> - <x>^2) and B = <y> - A<x>; the
    ratio form of B is computed independently and the two must agree.
    The defaults are the book's five-point worked data set.

    Parameters
    ----------
    x, y : array-like, optional
        Equal-length data vectors, n >= 2.

    Returns
    -------
    RichResult
        Keys: A, B, S.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs (6.42)-(6.49), (6.82).
    """
    x_d = [2.0, 3.0, 3.0, 5.0, 7.0] if x is None else x
    y_d = [1.0, 1.0, 3.0, 4.0, 6.0] if y is None else y
    A, B, S = _morin.least_squares_fit(x_d, y_d)
    payload = {"A": A, "B": B, "S": S}
    lines = [("A", A), ("B", B), ("S", S)]
    return RichResult(
        title="Least-squares line y = Ax + B, with the residual sum of squares.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "lsqfit: Least-squares line y = Ax + B. Morin (2016) eqs (6.42)-(6.49), (6.82)."
