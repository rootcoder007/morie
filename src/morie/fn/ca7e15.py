"""Level-2 model for the random intercept beta_0j = beta0 + u_0j.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_7_equation_15"]


def ca_chapter_7_equation_15(beta0, u_0j):
    """Level-2 model for the random intercept beta_0j = beta0 + u_0j

    Formula: beta_0j = beta_0 + u_0j

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.7 eq.7.15
    """
    value = float(beta0) + float(u_0j)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (7.15)"
    return RichResult(
        title='Level-2 model for the random intercept beta_0j = beta0 + u_0j',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca7e15: beta_0j = beta_0 + u_0j [Weisburd et al. 2022, eq. 7.15]'
