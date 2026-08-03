"""Nested-model F change test from the two R^2 values.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["f_nested_r2"]


def f_nested_r2(r2_full, r2_restricted, k_full, k_restricted, n):
    """Nested-model F change test from the two R^2 values

    Formula: F = ((R2_F - R2_R)/(k_F - k_R)) / ((1 - R2_F)/(n - k_F - 1))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'f' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.2 eq.2.19
    """
    payload = dict(_ca_crim.f_nested_r2(r2_full, r2_restricted, k_full, k_restricted, n))
    value = payload['f']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (2.19)"
    return RichResult(
        title='Nested-model F change test from the two R^2 values',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca2e19: F = ((R2_F - R2_R)/(k_F - k_R)) / ((1 - R2_F)/(n - k_F - 1)) [Weisburd et al. 2022, eq. 2.19]'
