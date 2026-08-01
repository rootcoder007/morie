"""Multiple logistic regression logit: b0 + sum(bk xk).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_4_equation_5"]


def ca_chapter_4_equation_5(b0, bs, xs):
    """Multiple logistic regression logit: b0 + sum(bk xk)

    Formula: logit(p) = b0 + b1 x1 + ... + bk xk

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.4 eq.4.5
    """
    value = _ca_crim.linear_predictor(b0, bs, xs)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (4.5)"
    return RichResult(
        title='Multiple logistic regression logit: b0 + sum(bk xk)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca4e5: logit(p) = b0 + b1 x1 + ... + bk xk [Weisburd et al. 2022, eq. 4.5]'
