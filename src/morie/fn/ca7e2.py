"""Fixed-effects dummy model y_i = beta0 + sum(beta_j x_j) + e_i.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_7_equation_2"]


def ca_chapter_7_equation_2(b0, bs, xs):
    """Fixed-effects dummy model y_i = beta0 + sum(beta_j x_j) + e_i

    Formula: y_i = beta0 + beta1 x1 + ... + beta_{j-1} x_{j-1} + e_i

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.7 eq.7.2
    """
    value = _ca_crim.linear_predictor(b0, bs, xs)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (7.2)"
    return RichResult(
        title='Fixed-effects dummy model y_i = beta0 + sum(beta_j x_j) + e_i',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca7e2: y_i = beta0 + beta1 x1 + ... + beta_{j-1} x_{j-1} + e_i [Weisburd et al. 2022, eq. 7.2]'
