"""Standard error of Hedges' g.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["se_g"]


def se_g(g, n1, n2):
    """Standard error of Hedges' g

    Formula: se_g = sqrt((n1+n2)/(n1 n2) + g^2/(2(n1+n2)))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.7
    """
    value = _ca_crim.se_g(g, n1, n2)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.7)"
    return RichResult(
        title="Standard error of Hedges' g",
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e7: se_g = sqrt((n1+n2)/(n1 n2) + g^2/(2(n1+n2))) [Weisburd et al. 2022, eq. 11.7]'
