"""Poisson model chi2 = (-2LLnull) - (-2LLfull).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_6_equation_5"]


def ca_chapter_6_equation_5(neg2ll_null, neg2ll_full):
    """Poisson model chi2 = (-2LLnull) - (-2LLfull)

    Formula: Model chi2 = (-2LL_null) - (-2LL_full)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.6 eq.6.5
    """
    value = _ca_crim.model_chi2(neg2ll_null, neg2ll_full)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (6.5)"
    return RichResult(
        title='Poisson model chi2 = (-2LLnull) - (-2LLfull)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca6e5: Model chi2 = (-2LL_null) - (-2LL_full) [Weisburd et al. 2022, eq. 6.5]'
