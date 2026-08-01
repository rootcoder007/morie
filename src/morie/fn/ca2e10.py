"""Confidence interval for an OLS coefficient.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_2_equation_10"]


def ca_chapter_2_equation_10(b, se, t_cv):
    """Confidence interval for an OLS coefficient

    Formula: b -/+ se * t_CV

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'lower' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.2 eq.2.10
    """
    payload = dict(_ca_crim.coef_ci(b, se, t_cv))
    value = payload['lower']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (2.10)"
    return RichResult(
        title='Confidence interval for an OLS coefficient',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca2e10: b -/+ se * t_CV [Weisburd et al. 2022, eq. 2.10]'
