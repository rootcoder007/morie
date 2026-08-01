"""Standard error of Fisher's Zr: 1/sqrt(n-3).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_11_equation_13"]


def ca_chapter_11_equation_13(n):
    """Standard error of Fisher's Zr: 1/sqrt(n-3)

    Formula: se_Zr = 1 / sqrt(n - 3)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.13
    """
    value = _ca_crim.se_fisher_z(n)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.13)"
    return RichResult(
        title="Standard error of Fisher's Zr: 1/sqrt(n-3)",
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e13: se_Zr = 1 / sqrt(n - 3) [Weisburd et al. 2022, eq. 11.13]'
