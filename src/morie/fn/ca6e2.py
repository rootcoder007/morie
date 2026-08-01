"""Poisson regression, population form ln(Y) = beta0 + beta1 x1.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_6_equation_2"]


def ca_chapter_6_equation_2(beta0, beta1, x1):
    """Poisson regression, population form ln(Y) = beta0 + beta1 x1

    Formula: ln(Y) = beta0 + beta1 x1

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.6 eq.6.2
    """
    value = _ca_crim.linear_predictor(beta0, [beta1], [x1])
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (6.2)"
    return RichResult(
        title='Poisson regression, population form ln(Y) = beta0 + beta1 x1',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca6e2: ln(Y) = beta0 + beta1 x1 [Weisburd et al. 2022, eq. 6.2]'
