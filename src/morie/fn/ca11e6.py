"""Independent t-test (composite of eqs 11.1-11.2).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_11_equation_6"]


def ca_chapter_11_equation_6(m1, m2, s1, s2, n1, n2):
    """Independent t-test (composite of eqs 11.1-11.2)

    Formula: t = (x1-x2) / (s_pooled sqrt((n1+n2)/(n1 n2)))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 't' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.6
    """
    payload = dict(_ca_crim.t_independent(m1, m2, s1, s2, n1, n2))
    value = payload['t']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.6)"
    return RichResult(
        title='Independent t-test (composite of eqs 11.1-11.2)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e6: t = (x1-x2) / (s_pooled sqrt((n1+n2)/(n1 n2))) [Weisburd et al. 2022, eq. 11.6]'
