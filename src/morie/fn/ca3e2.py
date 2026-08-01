"""Variance inflation factor VIF = 1 / (1 - R^2_x).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_3_equation_2"]


def ca_chapter_3_equation_2(r2_x):
    """Variance inflation factor VIF = 1 / (1 - R^2_x)

    Formula: VIF = 1/(1 - R^2_x) = 1/Tolerance

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.3 eq.3.2
    """
    value = _ca_crim.vif(r2_x)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (3.2)"
    return RichResult(
        title='Variance inflation factor VIF = 1 / (1 - R^2_x)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca3e2: VIF = 1/(1 - R^2_x) = 1/Tolerance [Weisburd et al. 2022, eq. 3.2]'
