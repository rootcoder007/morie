"""95% CI for a multinomial coefficient b -/+ 1.96 se.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_5_equation_5"]


def ca_chapter_5_equation_5(b, se):
    """95% CI for a multinomial coefficient b -/+ 1.96 se

    Formula: b -/+ 1.96 se

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'lower' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.5 eq.5.5
    """
    payload = dict(_ca_crim.coef_ci(b, se, 1.96))
    value = payload['lower']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (5.5)"
    return RichResult(
        title='95% CI for a multinomial coefficient b -/+ 1.96 se',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca5e5: b -/+ 1.96 se [Weisburd et al. 2022, eq. 5.5]'
