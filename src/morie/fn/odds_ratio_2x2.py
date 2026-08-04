"""Odds ratio OR = ad / bc.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["odds_ratio_2x2"]


def odds_ratio_2x2(a, b, c, d):
    """Odds ratio OR = ad / bc

    Formula: OR = (p1/(1-p1)) / (p2/(1-p2)) = ad/bc

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.10
    """
    value = _ca_crim.odds_ratio_2x2(a, b, c, d)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.10)"
    return RichResult(
        title='Odds ratio OR = ad / bc',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e10: OR = (p1/(1-p1)) / (p2/(1-p2)) = ad/bc [Weisburd et al. 2022, eq. 11.10]'


# compact alias per ledger/NAMING.md
oddsratio2x2 = odds_ratio_2x2
