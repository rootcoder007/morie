"""t-test for an individual regression coefficient.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_2_equation_9"]


def ca_chapter_2_equation_9(b, se):
    """t-test for an individual regression coefficient

    Formula: t = b / se_b

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.2 eq.2.9
    """
    value = _ca_crim.coef_t(b, se)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (2.9)"
    return RichResult(
        title='t-test for an individual regression coefficient',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca2e9: t = b / se_b [Weisburd et al. 2022, eq. 2.9]'
