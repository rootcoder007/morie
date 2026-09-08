"""Computational form of Q.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_11_equation_41"]


def ca_chapter_11_equation_41(ys, ws):
    """Computational form of Q

    Formula: Q = sum(w y^2) - (sum w y)^2 / sum w

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'q_computational' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.41
    """
    payload = dict(_ca_crim.q_statistic(ys, ws))
    value = payload['q_computational']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.41)"
    return RichResult(
        title='Computational form of Q',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e41: Q = sum(w y^2) - (sum w y)^2 / sum w [Weisburd et al. 2022, eq. 11.41]'
