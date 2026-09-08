"""Treatment coefficient controlling for one confounder.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["treatment_b_confounded"]


def treatment_b_confounded(r_yt, r_yx, r_tx, s_y, s_t):
    """Treatment coefficient controlling for one confounder

    Formula: b_t = ((r_yt - r_yx r_tx)/(1 - r_tx^2)) (s_y/s_t)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.9 eq.9.1
    """
    value = _ca_crim.treatment_b_confounded(r_yt, r_yx, r_tx, s_y, s_t)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (9.1)"
    return RichResult(
        title='Treatment coefficient controlling for one confounder',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca9e1: b_t = ((r_yt - r_yx r_tx)/(1 - r_tx^2)) (s_y/s_t) [Weisburd et al. 2022, eq. 9.1]'
