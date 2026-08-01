"""Rosenbaum-Rubin standardized absolute bias for PSM balance.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_10_equation_1"]


def ca_chapter_10_equation_1(mean_t, mean_c, s_t, s_c):
    """Rosenbaum-Rubin standardized absolute bias for PSM balance

    Formula: Bias = 100 (xbar_t - xbar_c) / sqrt((s_t^2 + s_c^2)/2)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.10 eq.10.1
    """
    value = _ca_crim.psm_standardized_bias(mean_t, mean_c, s_t, s_c)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (10.1)"
    return RichResult(
        title='Rosenbaum-Rubin standardized absolute bias for PSM balance',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca10e1: Bias = 100 (xbar_t - xbar_c) / sqrt((s_t^2 + s_c^2)/2) [Weisburd et al. 2022, eq. 10.1]'
