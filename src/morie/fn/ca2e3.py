"""OLS intercept b0 = ybar - b1 xbar.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_2_equation_3"]


def ca_chapter_2_equation_3(x, y):
    """OLS intercept b0 = ybar - b1 xbar

    Formula: b0 = ybar - b1 xbar

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'b0' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.2 eq.2.3
    """
    payload = dict(_ca_crim.ols_simple(x, y))
    value = payload['b0']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (2.3)"
    return RichResult(
        title='OLS intercept b0 = ybar - b1 xbar',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca2e3: b0 = ybar - b1 xbar [Weisburd et al. 2022, eq. 2.3]'
