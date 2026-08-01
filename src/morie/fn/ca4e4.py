"""Logit form: logit(p) = ln(p/(1-p)) = b0 + b1 x1.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_4_equation_4"]


def ca_chapter_4_equation_4(p):
    """Logit form: logit(p) = ln(p/(1-p)) = b0 + b1 x1

    Formula: logit(p) = ln(p/(1-p))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.4 eq.4.4
    """
    value = _ca_crim.logit(p)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (4.4)"
    return RichResult(
        title='Logit form: logit(p) = ln(p/(1-p)) = b0 + b1 x1',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca4e4: logit(p) = ln(p/(1-p)) [Weisburd et al. 2022, eq. 4.4]'
