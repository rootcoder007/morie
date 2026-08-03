"""OLS population model y = x beta + e (matrix fit).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ols_matrix"]


def ols_matrix(x, y):
    """OLS population model y = x beta + e (matrix fit)

    Formula: y = x beta + e

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.12 eq.12.3
    """
    payload = dict({"beta": _ca_crim.ols_matrix(x, y)['beta'].tolist(), "value": float(_ca_crim.ols_matrix(x, y)['beta'][-1])})
    value = payload['value']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (12.3)"
    return RichResult(
        title='OLS population model y = x beta + e (matrix fit)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca12e3: y = x beta + e [Weisburd et al. 2022, eq. 12.3]'
