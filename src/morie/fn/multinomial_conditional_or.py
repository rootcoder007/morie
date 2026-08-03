"""Conditional odds ratio OR_{m/n} = e^xbm / e^xbn.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["multinomial_conditional_or"]


def multinomial_conditional_or(xb_m, xb_n):
    """Conditional odds ratio OR_{m/n} = e^xbm / e^xbn

    Formula: Conditional OR_{m/n} = P(y=m)/P(y=n) = e^{xb_m - xb_n}

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.5 eq.5.4
    """
    value = _ca_crim.multinomial_conditional_or(xb_m, xb_n)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (5.4)"
    return RichResult(
        title='Conditional odds ratio OR_{m/n} = e^xbm / e^xbn',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca5e4: Conditional OR_{m/n} = P(y=m)/P(y=n) = e^{xb_m - xb_n} [Weisburd et al. 2022, eq. 5.4]'
