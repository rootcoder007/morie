"""Cumulative logit ln(P(y<=m)/P(y>m)).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_5_equation_7"]


def ca_chapter_5_equation_7(probs, m):
    """Cumulative logit ln(P(y<=m)/P(y>m))

    Formula: logit[P(y<=m)] = ln(P(y<=m)/P(y>m))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.5 eq.5.7
    """
    value = _ca_crim.cumulative_logit(probs, m)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (5.7)"
    return RichResult(
        title='Cumulative logit ln(P(y<=m)/P(y>m))',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca5e7: logit[P(y<=m)] = ln(P(y<=m)/P(y>m)) [Weisburd et al. 2022, eq. 5.7]'
