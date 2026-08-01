"""Independent-samples t-test written with explicit df.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_9_equation_11"]


def ca_chapter_9_equation_11(m1, m2, s1, s2, n1, n2):
    """Independent-samples t-test written with explicit df

    Formula: t = (x1-x2)/sqrt(((s1^2(n1-1)+s2^2(n2-1))/df)((n1+n2)/(n1 n2)))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 't' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.9 eq.9.11
    """
    payload = dict(_ca_crim.t_independent(m1, m2, s1, s2, n1, n2))
    value = payload['t']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (9.11)"
    return RichResult(
        title='Independent-samples t-test written with explicit df',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca9e11: t = (x1-x2)/sqrt(((s1^2(n1-1)+s2^2(n2-1))/df)((n1+n2)/(n1 n2))) [Weisburd et al. 2022, eq. 9.11]'
