"""Regression equation for the dummy = 0 subgroup.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_2_equation_21"]


def ca_chapter_2_equation_21(b0, bs, dummy_index):
    """Regression equation for the dummy = 0 subgroup

    Formula: y = b0 + b1 x1 + b2 x2 (dummy term drops out at 0)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'intercept' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.2 eq.2.21
    """
    payload = dict(_ca_crim.dummy_subgroup_equation(b0, bs, dummy_index, 0))
    value = payload['intercept']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (2.21)"
    return RichResult(
        title='Regression equation for the dummy = 0 subgroup',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca2e21: y = b0 + b1 x1 + b2 x2 (dummy term drops out at 0) [Weisburd et al. 2022, eq. 2.21]'
