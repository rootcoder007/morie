"""Wald statistic W = (b/se_b)^2.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_4_equation_15"]


def ca_chapter_4_equation_15(b, se):
    """Wald statistic W = (b/se_b)^2

    Formula: W = (b / se_b)^2

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.4 eq.4.15
    """
    value = _ca_crim.wald_statistic(b, se)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (4.15)"
    return RichResult(
        title='Wald statistic W = (b/se_b)^2',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca4e15: W = (b / se_b)^2 [Weisburd et al. 2022, eq. 4.15]'
