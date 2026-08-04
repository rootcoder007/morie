"""Likelihood ratio test chi2 = -2 (LL1 - LL2).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["lr_test_chi2"]


def lr_test_chi2(ll_null, ll_full):
    """Likelihood ratio test chi2 = -2 (LL1 - LL2)

    Formula: chi2 = -2 (LL_1 - LL_2)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.7 eq.7.8
    """
    value = _ca_crim.lr_test_chi2(ll_null, ll_full)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (7.8)"
    return RichResult(
        title='Likelihood ratio test chi2 = -2 (LL1 - LL2)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca7e8: chi2 = -2 (LL_1 - LL_2) [Weisburd et al. 2022, eq. 7.8]'


# compact alias per ledger/NAMING.md
lrtestchi2 = lr_test_chi2
