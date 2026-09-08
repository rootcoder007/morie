"""Higgins' I^2 = (Q - df)/Q x 100.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_11_equation_42"]


def ca_chapter_11_equation_42(q, df):
    """Higgins' I^2 = (Q - df)/Q x 100

    Formula: I^2 = ((Q - df) / Q) * 100

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.42
    """
    value = _ca_crim.i_squared(q, df)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.42)"
    return RichResult(
        title="Higgins' I^2 = (Q - df)/Q x 100",
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e42: I^2 = ((Q - df) / Q) * 100 [Weisburd et al. 2022, eq. 11.42]'
