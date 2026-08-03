"""Convert a risk ratio to an odds ratio.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["or_from_rr"]


def or_from_rr(rr, p2):
    """Convert a risk ratio to an odds ratio

    Formula: OR = RR p2 (1-p2) / [p2 (1 - RR p2)]

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.28
    """
    value = _ca_crim.or_from_rr(rr, p2)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.28)"
    return RichResult(
        title='Convert a risk ratio to an odds ratio',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e28: OR = RR p2 (1-p2) / [p2 (1 - RR p2)] [Weisburd et al. 2022, eq. 11.28]'
