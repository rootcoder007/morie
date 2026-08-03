"""Likelihood ratio chi2 = (-2LLreduced) - (-2LLfull).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["likelihood_ratio_chi2"]


def likelihood_ratio_chi2(neg2ll_reduced, neg2ll_full):
    """Likelihood ratio chi2 = (-2LLreduced) - (-2LLfull)

    Formula: LR chi2 = (-2LL_reduced) - (-2LL_full)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.4 eq.4.18
    """
    value = _ca_crim.likelihood_ratio_chi2(neg2ll_reduced, neg2ll_full)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (4.18)"
    return RichResult(
        title='Likelihood ratio chi2 = (-2LLreduced) - (-2LLfull)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca4e18: LR chi2 = (-2LL_reduced) - (-2LL_full) [Weisburd et al. 2022, eq. 4.18]'
