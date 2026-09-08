"""Cumulative probability P(y <= m) = sum_{j<=m} P(y=j).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["cumulative_probability"]


def cumulative_probability(probs, m):
    """Cumulative probability P(y <= m) = sum_{j<=m} P(y=j)

    Formula: P(y<=m) = sum_{j=1..m} P(y=j)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.5 eq.5.6
    """
    value = _ca_crim.cumulative_probability(probs, m)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (5.6)"
    return RichResult(
        title='Cumulative probability P(y <= m) = sum_{j<=m} P(y=j)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca5e6: P(y<=m) = sum_{j=1..m} P(y=j) [Weisburd et al. 2022, eq. 5.6]'
