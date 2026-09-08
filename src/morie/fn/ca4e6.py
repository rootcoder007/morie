"""Convert a logit to a probability: p = e^logit/(1+e^logit).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_4_equation_6"]


def ca_chapter_4_equation_6(logit_value):
    """Convert a logit to a probability: p = e^logit/(1+e^logit)

    Formula: p = e^logit / (1 + e^logit)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.4 eq.4.6
    """
    value = _ca_crim.inv_logit(logit_value)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (4.6)"
    return RichResult(
        title='Convert a logit to a probability: p = e^logit/(1+e^logit)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca4e6: p = e^logit / (1 + e^logit) [Weisburd et al. 2022, eq. 4.6]'
