"""Standard error of the mean effect size 1/sqrt(sum w).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_11_equation_36"]


def ca_chapter_11_equation_36(ys, ws):
    """Standard error of the mean effect size 1/sqrt(sum w)

    Formula: se_ybar = sqrt(1 / sum(w_i))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'se' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.36
    """
    payload = dict(_ca_crim.mean_effect_size(ys, ws))
    value = payload['se']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.36)"
    return RichResult(
        title='Standard error of the mean effect size 1/sqrt(sum w)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e36: se_ybar = sqrt(1 / sum(w_i)) [Weisburd et al. 2022, eq. 11.36]'
