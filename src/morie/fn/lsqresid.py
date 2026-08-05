"""Normal equation dS/dB = 0: the residuals sum to zero at the optimum.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (6.92).
"""

from . import _array_core as np

from . import _morin

from ._richresult import RichResult

__all__ = ["lsqresid"]


def lsqresid(x, y):
    """Normal equation dS/dB = 0: the residuals sum to zero at the optimum.

    Parameters
    ----------
    x, y : array-like
        Equal-length data vectors, n >= 2.

    Returns
    -------
    RichResult
        Keys: residual_sum, A, B.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (6.92).
    """
    A, B, S = _morin.least_squares_fit(x, y)
    x_a = np.atleast_1d(np.asarray(x, dtype=float))
    y_a = np.atleast_1d(np.asarray(y, dtype=float))
    resid_sum = float(np.sum(y_a - (A * x_a + B)))
    payload = {"residual_sum": resid_sum, "A": A, "B": B}
    lines = [("sum of residuals", resid_sum)]
    return RichResult(
        title="Normal equation dS/dB = 0: the residuals sum to zero at the optimum.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "lsqresid: The least-squares residuals sum to zero. Morin (2016) eq (6.92)."
