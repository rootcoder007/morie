"""Noncentrality delta for a correlation coefficient.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_8_equation_6"]


def ca_chapter_8_equation_6(r, n):
    """Noncentrality delta for a correlation coefficient

    Formula: delta = r sqrt(n-2) / sqrt(1 - r^2)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.8 eq.8.6
    """
    value = _ca_crim.noncentrality_delta_r(r, n)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (8.6)"
    return RichResult(
        title='Noncentrality delta for a correlation coefficient',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca8e6: delta = r sqrt(n-2) / sqrt(1 - r^2) [Weisburd et al. 2022, eq. 8.6]'
