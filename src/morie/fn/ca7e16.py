"""Level-2 model for the random slope beta_1j = beta1 + u_1j.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_7_equation_16"]


def ca_chapter_7_equation_16(beta1, u_1j):
    """Level-2 model for the random slope beta_1j = beta1 + u_1j

    Formula: beta_1j = beta_1 + u_1j

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.7 eq.7.16
    """
    value = float(beta1) + float(u_1j)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (7.16)"
    return RichResult(
        title='Level-2 model for the random slope beta_1j = beta1 + u_1j',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca7e16: beta_1j = beta_1 + u_1j [Weisburd et al. 2022, eq. 7.16]'
