"""Covariance of zero-mean variables: Cov(X,Y) = E(XY).

Morin (2016), Probability: For the Enthusiastic Beginner, eq (6.8).
"""

from . import _array_core as np

from . import _morin

from ._richresult import RichResult

__all__ = ["covzmean"]


def covzmean(x, y):
    """Covariance of zero-mean variables: Cov(X,Y) = E(XY).

    Refuses data whose means are not zero: use the shortcut form
    instead.

    Parameters
    ----------
    x, y : array-like
        Equal-length zero-mean data vectors.

    Returns
    -------
    RichResult
        Keys: cov.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (6.8).
    """
    x_a = np.atleast_1d(np.asarray(x, dtype=float))
    y_a = np.atleast_1d(np.asarray(y, dtype=float))
    if abs(float(x_a.mean())) > 1e-9 or abs(float(y_a.mean())) > 1e-9:
        raise ValueError("this form needs zero-mean data; use eq (6.14) otherwise")
    value = float(np.mean(x_a * y_a))
    payload = {"cov": value}
    lines = [("E(XY)", value)]
    return RichResult(
        title="Covariance of zero-mean variables: Cov(X,Y) = E(XY).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "covzmean: Cov(X,Y) = E(XY) for zero-mean data. Morin (2016) eq (6.8)."
