"""Hedges' g = J d.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_11_equation_4"]


def ca_chapter_11_equation_4(d, n1, n2):
    """Hedges' g = J d

    Formula: g = J d

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.4
    """
    value = _ca_crim.hedges_g(d, n1, n2)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.4)"
    return RichResult(
        title="Hedges' g = J d",
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e4: g = J d [Weisburd et al. 2022, eq. 11.4]'
