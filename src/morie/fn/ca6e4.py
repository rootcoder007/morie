"""Poisson model for rearrests ln(y_i) = b0 + b1 x_i.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_6_equation_4"]


def ca_chapter_6_equation_4(b0, b1, x1):
    """Poisson model for rearrests ln(y_i) = b0 + b1 x_i

    Formula: ln(y_i) = b0 + b1 x_i (+ e_i)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.6 eq.6.4
    """
    value = _ca_crim.poisson_loglink_predict(b0, b1, x1)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (6.4)"
    return RichResult(
        title='Poisson model for rearrests ln(y_i) = b0 + b1 x_i',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca6e4: ln(y_i) = b0 + b1 x_i (+ e_i) [Weisburd et al. 2022, eq. 6.4]'
