"""Ordinal logit, additive form tau_m + Xb.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_5_equation_8"]


def ca_chapter_5_equation_8(tau_m, bs, xs):
    """Ordinal logit, additive form tau_m + Xb

    Formula: logit(y<=m|x) = tau_m + b1 x1 + ... + bk xk

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.5 eq.5.8
    """
    value = _ca_crim.ordinal_logit(tau_m, bs, xs, 'plus')
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (5.8)"
    return RichResult(
        title='Ordinal logit, additive form tau_m + Xb',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca5e8: logit(y<=m|x) = tau_m + b1 x1 + ... + bk xk [Weisburd et al. 2022, eq. 5.8]'
