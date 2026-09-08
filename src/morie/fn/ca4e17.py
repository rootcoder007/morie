"""95% confidence interval b -/+ 1.96 se.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_4_equation_17"]


def ca_chapter_4_equation_17(b, se):
    """95% confidence interval b -/+ 1.96 se

    Formula: b -/+ 1.96 se

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'lower' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.4 eq.4.17
    """
    payload = dict(_ca_crim.coef_ci(b, se, 1.96))
    value = payload['lower']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (4.17)"
    return RichResult(
        title='95% confidence interval b -/+ 1.96 se',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca4e17: b -/+ 1.96 se [Weisburd et al. 2022, eq. 4.17]'
