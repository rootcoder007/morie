"""Simple linear regression model y = b0 + b1 x + e (fit).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_2_equation_1"]


def ca_chapter_2_equation_1(x, y):
    """Simple linear regression model y = b0 + b1 x + e (fit)

    Formula: y = b0 + b1 x + e

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'b1' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.2 eq.2.1
    """
    payload = dict(_ca_crim.ols_simple(x, y))
    value = payload['b1']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (2.1)"
    return RichResult(
        title='Simple linear regression model y = b0 + b1 x + e (fit)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca2e1: y = b0 + b1 x + e [Weisburd et al. 2022, eq. 2.1]'
