"""Point-biserial r to d: d = 2r / sqrt(1-r^2).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["d_from_r_pointbiserial"]


def d_from_r_pointbiserial(r):
    """Point-biserial r to d: d = 2r / sqrt(1-r^2)

    Formula: d = 2r / sqrt(1 - r^2)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.22
    """
    value = _ca_crim.d_from_r_pointbiserial(r)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.22)"
    return RichResult(
        title='Point-biserial r to d: d = 2r / sqrt(1-r^2)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e22: d = 2r / sqrt(1 - r^2) [Weisburd et al. 2022, eq. 11.22]'
