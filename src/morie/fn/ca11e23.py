"""se_d from se_r: sqrt(4 se_r^2 / (1-r^2)^3).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_11_equation_23"]


def ca_chapter_11_equation_23(r, se_r):
    """se_d from se_r: sqrt(4 se_r^2 / (1-r^2)^3)

    Formula: se_d = sqrt(4 se_r^2 / (1 - r^2)^3)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.23
    """
    value = _ca_crim.se_d_from_se_r(r, se_r)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.23)"
    return RichResult(
        title='se_d from se_r: sqrt(4 se_r^2 / (1-r^2)^3)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e23: se_d = sqrt(4 se_r^2 / (1 - r^2)^3) [Weisburd et al. 2022, eq. 11.23]'
