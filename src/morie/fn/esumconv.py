"""E(X + Y) from the convolution equals E(X) + E(Y).

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (3.11)-(3.12).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["esumconv"]


def esumconv(values_x, probs_x, values_y, probs_y):
    """E(X + Y) from the convolution equals E(X) + E(Y).

    Parameters
    ----------
    values_x, probs_x : array-like
        The pmf of X; probs must be >= 0 and sum to 1.
    values_y, probs_y : array-like
        The pmf of Y, assumed independent of X.

    Returns
    -------
    RichResult
        Keys: e_sum, e_x_plus_e_y.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs (3.11)-(3.12).
    """
    values, probs = _morin.pmf_sum_convolution(values_x, probs_x, values_y, probs_y)
    e_sum = _morin.pmf_expectation(values, probs)
    e_parts = (_morin.pmf_expectation(values_x, probs_x)
               + _morin.pmf_expectation(values_y, probs_y))
    if abs(e_sum - e_parts) > 1e-9:
        raise AssertionError("E(X+Y) != E(X) + E(Y)")
    payload = {"e_sum": e_sum, "e_x_plus_e_y": e_parts}
    lines = [("E(X+Y)", e_sum)]
    return RichResult(
        title="E(X + Y) from the convolution equals E(X) + E(Y).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "esumconv: E(X+Y) from the convolved pmf equals E(X)+E(Y). Morin (2016) eqs (3.11)-(3.12)."
