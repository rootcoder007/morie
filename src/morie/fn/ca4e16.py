"""z-test for a logistic coefficient z = b/se_b.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_4_equation_16"]


def ca_chapter_4_equation_16(b, se):
    """z-test for a logistic coefficient z = b/se_b

    Formula: z = b / se_b

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.4 eq.4.16
    """
    value = _ca_crim.coef_t(b, se)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (4.16)"
    return RichResult(
        title='z-test for a logistic coefficient z = b/se_b',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca4e16: z = b / se_b [Weisburd et al. 2022, eq. 4.16]'
