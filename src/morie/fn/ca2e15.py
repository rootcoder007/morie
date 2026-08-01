"""Adjusted R^2 penalizing model complexity.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_2_equation_15"]


def ca_chapter_2_equation_15(r2, n, k):
    """Adjusted R^2 penalizing model complexity

    Formula: Adj R^2 = 1 - (1-R^2)(n-1)/(n-k-1)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.2 eq.2.15
    """
    value = _ca_crim.adjusted_r2(r2, n, k)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (2.15)"
    return RichResult(
        title='Adjusted R^2 penalizing model complexity',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca2e15: Adj R^2 = 1 - (1-R^2)(n-1)/(n-k-1) [Weisburd et al. 2022, eq. 2.15]'
