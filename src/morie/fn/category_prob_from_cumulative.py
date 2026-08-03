"""Category probability pi_j = P(Y <= j) - P(Y <= j-1).

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["category_prob_from_cumulative"]


def category_prob_from_cumulative(cum_probs, j):
    """Category probability pi_j = P(Y <= j) - P(Y <= j-1)

    Formula: pi_j = P(Y <= j) - P(Y <= j-1)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (3.12).
    """
    value = _acd.category_prob_from_cumulative(cum_probs, j)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (3.12)"
    return RichResult(
        title='Category probability pi_j = P(Y <= j) - P(Y <= j-1)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '3e12: pi_j = P(Y <= j) - P(Y <= j-1) [Bilder & Loughin 2025, eq. 3.12]'
