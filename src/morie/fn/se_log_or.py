"""Standard error of the logged odds ratio.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["se_log_or"]


def se_log_or(a, b, c, d):
    """Standard error of the logged odds ratio

    Formula: se_lnOR = sqrt(1/a + 1/b + 1/c + 1/d)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.11
    """
    value = _ca_crim.se_log_or(a, b, c, d)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.11)"
    return RichResult(
        title='Standard error of the logged odds ratio',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e11: se_lnOR = sqrt(1/a + 1/b + 1/c + 1/d) [Weisburd et al. 2022, eq. 11.11]'
