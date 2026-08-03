"""se_d for the probit method.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["se_d_probit"]


def se_d_probit(p1, p2, n1, n2):
    """se_d for the probit method

    Formula: se_d = sqrt(2 pi p1(1-p1) e^{z1^2}/n1 + 2 pi p2(1-p2) e^{z2^2}/n2)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.21
    """
    value = _ca_crim.se_d_probit(p1, p2, n1, n2)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.21)"
    return RichResult(
        title='se_d for the probit method',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e21: se_d = sqrt(2 pi p1(1-p1) e^{z1^2}/n1 + 2 pi p2(1-p2) e^{z2^2}/n2) [Weisburd et al. 2022, eq. 11.21]'
