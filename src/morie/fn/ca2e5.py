"""t-test for the OLS slope (residual-variance form).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_2_equation_5"]


def ca_chapter_2_equation_5(x, y):
    """t-test for the OLS slope (residual-variance form)

    Formula: t = b1 / sqrt((sum(y-yhat)^2/(n-2)) / sum(x-xbar)^2)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 't' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.2 eq.2.5
    """
    payload = dict(_ca_crim.ols_simple(x, y))
    value = payload['t']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (2.5)"
    return RichResult(
        title='t-test for the OLS slope (residual-variance form)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca2e5: t = b1 / sqrt((sum(y-yhat)^2/(n-2)) / sum(x-xbar)^2) [Weisburd et al. 2022, eq. 2.5]'
