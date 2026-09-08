"""Multinomial probability P(y=m) = e^xbm / sum_j e^xbj.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["multinomial_probs"]


def multinomial_probs(xbs):
    """Multinomial probability P(y=m) = e^xbm / sum_j e^xbj

    Formula: P(y=m) = e^{xb_m} / sum_j e^{xb_j}

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.5 eq.5.3
    """
    payload = dict({"probs": _ca_crim.multinomial_probs(xbs).tolist(), "value": float(_ca_crim.multinomial_probs(xbs)[0])})
    value = payload['value']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (5.3)"
    return RichResult(
        title='Multinomial probability P(y=m) = e^xbm / sum_j e^xbj',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca5e3: P(y=m) = e^{xb_m} / sum_j e^{xb_j} [Weisburd et al. 2022, eq. 5.3]'
