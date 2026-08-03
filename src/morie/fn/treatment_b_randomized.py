"""Treatment coefficient under randomization b_t = r_yt s_y/s_t.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["treatment_b_randomized"]


def treatment_b_randomized(r_yt, s_y, s_t):
    """Treatment coefficient under randomization b_t = r_yt s_y/s_t

    Formula: b_t = r_yt (s_y / s_t)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.9 eq.9.2
    """
    value = _ca_crim.treatment_b_randomized(r_yt, s_y, s_t)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (9.2)"
    return RichResult(
        title='Treatment coefficient under randomization b_t = r_yt s_y/s_t',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca9e2: b_t = r_yt (s_y / s_t) [Weisburd et al. 2022, eq. 9.2]'
