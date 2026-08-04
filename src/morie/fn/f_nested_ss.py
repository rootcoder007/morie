"""Nested-model F change test from residual sums of squares.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["f_nested_ss"]


def f_nested_ss(ss_resid_restricted, ss_resid_full, k_full, k_restricted, n):
    """Nested-model F change test from residual sums of squares

    Formula: F = (SS_resid(R) - SS_resid(F)) / MS_resid(F)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'f' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.2 eq.2.18
    """
    payload = dict(_ca_crim.f_nested_ss(ss_resid_restricted, ss_resid_full, k_full, k_restricted, n))
    value = payload['f']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (2.18)"
    return RichResult(
        title='Nested-model F change test from residual sums of squares',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca2e18: F = (SS_resid(R) - SS_resid(F)) / MS_resid(F) [Weisburd et al. 2022, eq. 2.18]'


# compact alias per ledger/NAMING.md
fnestedss = f_nested_ss
