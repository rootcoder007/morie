"""Truncated power cubic spline.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["truncated_power_spline"]


def truncated_power_spline(x, betas, knots):
    """Truncated power cubic spline

    Formula: f(x) = b0 + b1 x + b2 x^2 + b3 x^3 + sum_d b_(3+d) (x - k_d)^3_+

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (6.36).
    """
    value = _acd.truncated_power_spline(x, betas, knots)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (6.36)"
    return RichResult(
        title='Truncated power cubic spline',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '6e36: f(x) = b0 + b1 x + b2 x^2 + b3 x^3 + sum_d b_(3+d) (x - k_d)^3_+ [Bilder & Loughin 2025, eq. 6.36]'
