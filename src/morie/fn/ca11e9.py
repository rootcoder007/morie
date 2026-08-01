"""Standard error of the logged risk ratio.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_11_equation_9"]


def ca_chapter_11_equation_9(p1, p2, n1, n2):
    """Standard error of the logged risk ratio

    Formula: se_lnRR = sqrt((1-p1)/(n1 p1) + (1-p2)/(n2 p2))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.9
    """
    value = _ca_crim.se_log_rr(p1, p2, n1, n2)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.9)"
    return RichResult(
        title='Standard error of the logged risk ratio',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e9: se_lnRR = sqrt((1-p1)/(n1 p1) + (1-p2)/(n2 p2)) [Weisburd et al. 2022, eq. 11.9]'
