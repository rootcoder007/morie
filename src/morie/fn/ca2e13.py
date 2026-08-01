"""Residual variance sigma^2_residual.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_2_equation_13"]


def ca_chapter_2_equation_13(y, yhat):
    """Residual variance sigma^2_residual

    Formula: sigma^2_residual = sum(yi - yhat_i)^2 / n

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'var_resid' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.2 eq.2.13
    """
    payload = dict(_ca_crim.variance_partition(y, yhat))
    value = payload['var_resid']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (2.13)"
    return RichResult(
        title='Residual variance sigma^2_residual',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca2e13: sigma^2_residual = sum(yi - yhat_i)^2 / n [Weisburd et al. 2022, eq. 2.13]'
