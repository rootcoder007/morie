"""Standardized regression coefficient Beta = b s_x / s_y.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["beta_standardized"]


def beta_standardized(b, s_x, s_y):
    """Standardized regression coefficient Beta = b s_x / s_y

    Formula: Beta = b (s_x / s_y)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.2 eq.2.20
    """
    value = _ca_crim.beta_standardized(b, s_x, s_y)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (2.20)"
    return RichResult(
        title='Standardized regression coefficient Beta = b s_x / s_y',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca2e20: Beta = b (s_x / s_y) [Weisburd et al. 2022, eq. 2.20]'
