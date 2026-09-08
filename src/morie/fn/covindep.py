"""Independence makes the covariance vanish.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (6.63).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["covindep"]


def covindep(x, y, tol=1e-09):
    """Independence makes the covariance vanish.

    ``near_zero`` is the decision |Cov| <= tol.  A zero covariance is
    necessary but not sufficient for independence.

    Parameters
    ----------
    x, y : array-like
        Equal-length data vectors, n >= 2.
    tol : float
        Absolute tolerance on the covariance.

    Returns
    -------
    RichResult
        Keys: cov, near_zero.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (6.63).
    """
    cov = _morin.cov_shortcut(x, y)
    payload = {"cov": cov, "near_zero": abs(cov) <= float(tol)}
    lines = [("Cov", cov)]
    return RichResult(
        title="Independence makes the covariance vanish.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "covindep: Independence implies a vanishing covariance. Morin (2016) eq (6.63)."
