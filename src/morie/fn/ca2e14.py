"""R^2 = SS_model / SS_total.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_2_equation_14"]


def ca_chapter_2_equation_14(y, yhat):
    """R^2 = SS_model / SS_total

    Formula: R^2 = sum(yhat-ybar)^2 / sum(y-ybar)^2

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'r2' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.2 eq.2.14
    """
    payload = dict(_ca_crim.variance_partition(y, yhat))
    value = payload['r2']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (2.14)"
    return RichResult(
        title='R^2 = SS_model / SS_total',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca2e14: R^2 = sum(yhat-ybar)^2 / sum(y-ybar)^2 [Weisburd et al. 2022, eq. 2.14]'
