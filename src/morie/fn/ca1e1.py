"""GLM structural model: predicted value Yhat = b0 + sum(bk xk).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_1_equation_1"]


def ca_chapter_1_equation_1(b0, bs, xs):
    """GLM structural model: predicted value Yhat = b0 + sum(bk xk)

    Formula: Yhat = b0 + b1 x1 + ... + bk xk

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.1 eq.1.1
    """
    value = _ca_crim.linear_predictor(b0, bs, xs)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (1.1)"
    return RichResult(
        title='GLM structural model: predicted value Yhat = b0 + sum(bk xk)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca1e1: Yhat = b0 + b1 x1 + ... + bk xk [Weisburd et al. 2022, eq. 1.1]'
