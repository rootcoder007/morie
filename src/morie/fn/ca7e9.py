"""Random intercept model with fixed slopes.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_7_equation_9"]


def ca_chapter_7_equation_9(b0, bs, xs, u_j):
    """Random intercept model with fixed slopes

    Formula: y_ij = beta0 + beta1 x1 + ... + beta_k x_k + u_j + e_ij

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.7 eq.7.9
    """
    value = _ca_crim.multilevel_predict(b0, bs, xs, [u_j])
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (7.9)"
    return RichResult(
        title='Random intercept model with fixed slopes',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca7e9: y_ij = beta0 + beta1 x1 + ... + beta_k x_k + u_j + e_ij [Weisburd et al. 2022, eq. 7.9]'
