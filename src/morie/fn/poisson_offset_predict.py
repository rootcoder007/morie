"""Poisson regression with an offset (exposure) term.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["poisson_offset_predict"]


def poisson_offset_predict(b0, b1, x1, exposure):
    """Poisson regression with an offset (exposure) term

    Formula: ln(y) = b0 + b1 x1 + offset(ln exposure)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.6 eq.6.7
    """
    value = _ca_crim.poisson_offset_predict(b0, b1, x1, exposure)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (6.7)"
    return RichResult(
        title='Poisson regression with an offset (exposure) term',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca6e7: ln(y) = b0 + b1 x1 + offset(ln exposure) [Weisburd et al. 2022, eq. 6.7]'
