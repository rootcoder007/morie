"""Grand-mean model y_i = beta0 + e_i.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_7_equation_1"]


def ca_chapter_7_equation_1(y):
    """Grand-mean model y_i = beta0 + e_i

    Formula: y_i = beta0 + e_i; beta0 = mean(y)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'intercept' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.7 eq.7.1
    """
    payload = dict(_ca_crim.grand_mean_model(y))
    value = payload['intercept']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (7.1)"
    return RichResult(
        title='Grand-mean model y_i = beta0 + e_i',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca7e1: y_i = beta0 + e_i; beta0 = mean(y) [Weisburd et al. 2022, eq. 7.1]'
