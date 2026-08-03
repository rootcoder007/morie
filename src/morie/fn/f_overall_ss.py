"""Overall model F-test from sums of squares.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["f_overall_ss"]


def f_overall_ss(ss_model, ss_resid, n, k):
    """Overall model F-test from sums of squares

    Formula: F = (SS_model/df_model) / (SS_resid/df_resid)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'f' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.2 eq.2.16
    """
    payload = dict(_ca_crim.f_overall_ss(ss_model, ss_resid, n, k))
    value = payload['f']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (2.16)"
    return RichResult(
        title='Overall model F-test from sums of squares',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca2e16: F = (SS_model/df_model) / (SS_resid/df_resid) [Weisburd et al. 2022, eq. 2.16]'
