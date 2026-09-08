"""Covariance shortcut: Cov = E(XY) - mu_x mu_y.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (6.14).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["covshort"]


def covshort(x, y):
    """Covariance shortcut: Cov = E(XY) - mu_x mu_y.

    Cross-checked against the deviation form.

    Parameters
    ----------
    x, y : array-like
        Equal-length data vectors, n >= 2.

    Returns
    -------
    RichResult
        Keys: cov.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (6.14).
    """
    value = _morin.cov_shortcut(x, y)
    payload = {"cov": value}
    lines = [("Cov(x, y)", value)]
    return RichResult(
        title="Covariance shortcut: Cov = E(XY) - mu_x mu_y.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "covshort: Cov = mean(xy) - mean(x) mean(y). Morin (2016) eq (6.14)."
