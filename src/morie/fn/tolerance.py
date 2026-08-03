"""Tolerance = 1 - R^2_x (multicollinearity diagnostic).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["tolerance"]


def tolerance(r2_x):
    """Tolerance = 1 - R^2_x (multicollinearity diagnostic)

    Formula: Tolerance = 1 - R^2_x

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.3 eq.3.1
    """
    value = _ca_crim.tolerance(r2_x)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (3.1)"
    return RichResult(
        title='Tolerance = 1 - R^2_x (multicollinearity diagnostic)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca3e1: Tolerance = 1 - R^2_x [Weisburd et al. 2022, eq. 3.1]'
