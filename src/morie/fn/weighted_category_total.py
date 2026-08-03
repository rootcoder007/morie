"""Survey-weighted population category count.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["weighted_category_total"]


def weighted_category_total(weights, ys, category):
    """Survey-weighted population category count

    Formula: N_hat_i = sum_s w_s I(y_s = i)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (6.7).
    """
    value = _acd.weighted_category_total(weights, ys, category)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (6.7)"
    return RichResult(
        title='Survey-weighted population category count',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '6e7: N_hat_i = sum_s w_s I(y_s = i) [Bilder & Loughin 2025, eq. 6.7]'
