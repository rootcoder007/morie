"""Odds ratio for a one-unit change: OR = e^b.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["odds_ratio_unit_change"]


def odds_ratio_unit_change(b):
    """Odds ratio for a one-unit change: OR = e^b

    Formula: OR = odds(x+1) / odds(x) = e^b

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.4 eq.4.8
    """
    value = _ca_crim.odds_ratio_unit_change(b)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (4.8)"
    return RichResult(
        title='Odds ratio for a one-unit change: OR = e^b',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca4e8: OR = odds(x+1) / odds(x) = e^b [Weisburd et al. 2022, eq. 4.8]'
