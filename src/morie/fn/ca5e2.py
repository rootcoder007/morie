"""Multinomial logit of category m vs reference category.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_5_equation_2"]


def ca_chapter_5_equation_2(xb_m):
    """Multinomial logit of category m vs reference category

    Formula: logit(y=m|x) = ln(P(y=m|x)/P(y=1|x)) = xb_m

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.5 eq.5.2
    """
    value = float(xb_m)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (5.2)"
    return RichResult(
        title='Multinomial logit of category m vs reference category',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca5e2: logit(y=m|x) = ln(P(y=m|x)/P(y=1|x)) = xb_m [Weisburd et al. 2022, eq. 5.2]'
