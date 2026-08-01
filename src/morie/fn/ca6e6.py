"""Worked model chi-square for the rearrest Poisson model.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_6_equation_6"]


def ca_chapter_6_equation_6(neg2ll_null, neg2ll_full):
    """Worked model chi-square for the rearrest Poisson model

    Formula: Model chi2 = (-2LL_null) - (-2LL_full)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.6 eq.6.6
    """
    value = _ca_crim.model_chi2(neg2ll_null, neg2ll_full)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (6.6)"
    return RichResult(
        title='Worked model chi-square for the rearrest Poisson model',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca6e6: Model chi2 = (-2LL_null) - (-2LL_full) [Weisburd et al. 2022, eq. 6.6]'
