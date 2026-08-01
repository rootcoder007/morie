"""Random-effects weight w = 1/(se^2 + tau^2).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_11_equation_43"]


def ca_chapter_11_equation_43(se, tau2):
    """Random-effects weight w = 1/(se^2 + tau^2)

    Formula: w_i = 1 / (se_i^2 + tau^2)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.43
    """
    value = _ca_crim.random_effects_weight(se, tau2)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.43)"
    return RichResult(
        title='Random-effects weight w = 1/(se^2 + tau^2)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e43: w_i = 1 / (se_i^2 + tau^2) [Weisburd et al. 2022, eq. 11.43]'
