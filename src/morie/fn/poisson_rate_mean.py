"""Poisson rate regression with an offset.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["poisson_rate_mean"]


def poisson_rate_mean(b0, bs, xs, exposure):
    """Poisson rate regression with an offset

    Formula: log(mu) = log(t) + b0 + b1 x1 + ... + bp xp

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (4.15).
    """
    value = _acd.poisson_rate_mean(b0, bs, xs, exposure)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (4.15)"
    return RichResult(
        title='Poisson rate regression with an offset',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '4e15: log(mu) = log(t) + b0 + b1 x1 + ... + bp xp [Bilder & Loughin 2025, eq. 4.15]'
