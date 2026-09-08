"""Q_between = Q - Q_within.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_11_equation_46"]


def ca_chapter_11_equation_46(ys_by_group, ws_by_group):
    """Q_between = Q - Q_within

    Formula: Q_between = Q_total - Q_within

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'q_between' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.46
    """
    payload = dict(_ca_crim.q_within_between(ys_by_group, ws_by_group))
    value = payload['q_between']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.46)"
    return RichResult(
        title='Q_between = Q - Q_within',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e46: Q_between = Q_total - Q_within [Weisburd et al. 2022, eq. 11.46]'
