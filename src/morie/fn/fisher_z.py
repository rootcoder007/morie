"""Fisher's Z transformation Zr = 0.5 ln((1+r)/(1-r)).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["fisher_z"]


def fisher_z(r):
    """Fisher's Z transformation Zr = 0.5 ln((1+r)/(1-r))

    Formula: Zr = (1/2) ln((1+r)/(1-r))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.12
    """
    value = _ca_crim.fisher_z(r)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.12)"
    return RichResult(
        title="Fisher's Z transformation Zr = 0.5 ln((1+r)/(1-r))",
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e12: Zr = (1/2) ln((1+r)/(1-r)) [Weisburd et al. 2022, eq. 11.12]'
