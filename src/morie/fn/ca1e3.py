"""Logit link: logit(pi) = ln(pi / (1 - pi)).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_1_equation_3"]


def ca_chapter_1_equation_3(p):
    """Logit link: logit(pi) = ln(pi / (1 - pi))

    Formula: logit(pi) = ln(pi / (1 - pi))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.1 eq.1.3
    """
    value = _ca_crim.logit(p)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (1.3)"
    return RichResult(
        title='Logit link: logit(pi) = ln(pi / (1 - pi))',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca1e3: logit(pi) = ln(pi / (1 - pi)) [Weisburd et al. 2022, eq. 1.3]'
