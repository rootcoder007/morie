"""Piecewise cubic around a knot.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["piecewise_cubic"]


def piecewise_cubic(x, knot, coef_left, coef_right):
    """Piecewise cubic around a knot

    Formula: f(x) = delta-cubic for x <= k; gamma-cubic for x > k

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (6.34).
    """
    value = _acd.piecewise_cubic(x, knot, coef_left, coef_right)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (6.34)"
    return RichResult(
        title='Piecewise cubic around a knot',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '6e34: f(x) = delta-cubic for x <= k; gamma-cubic for x > k [Bilder & Loughin 2025, eq. 6.34]'


# compact alias per ledger/NAMING.md
piecewisecubic = piecewise_cubic
