"""Level-2 model for cluster intercepts beta0j = beta00 + u_j.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_7_equation_5"]


def ca_chapter_7_equation_5(beta00, u_j):
    """Level-2 model for cluster intercepts beta0j = beta00 + u_j

    Formula: beta_0j = beta_00 + u_j

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.7 eq.7.5
    """
    value = float(beta00) + float(u_j)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (7.5)"
    return RichResult(
        title='Level-2 model for cluster intercepts beta0j = beta00 + u_j',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca7e5: beta_0j = beta_00 + u_j [Weisburd et al. 2022, eq. 7.5]'
