"""t-test for the slope from the correlation coefficient.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_2_equation_6"]


def ca_chapter_2_equation_6(x, y):
    """t-test for the slope from the correlation coefficient

    Formula: t = r sqrt((n-2)/(1-r^2))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 't_from_r' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.2 eq.2.6
    """
    payload = dict(_ca_crim.ols_simple(x, y))
    value = payload['t_from_r']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (2.6)"
    return RichResult(
        title='t-test for the slope from the correlation coefficient',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca2e6: t = r sqrt((n-2)/(1-r^2)) [Weisburd et al. 2022, eq. 2.6]'
