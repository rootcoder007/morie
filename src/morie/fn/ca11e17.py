"""se_d from se_lnOR, logit method.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_11_equation_17"]


def ca_chapter_11_equation_17(se_ln_or):
    """se_d from se_lnOR, logit method

    Formula: se_d = sqrt(se_lnOR^2 / (pi^2/3))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.17
    """
    value = _ca_crim.se_d_from_se_log_or(se_ln_or, 'logit')
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.17)"
    return RichResult(
        title='se_d from se_lnOR, logit method',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e17: se_d = sqrt(se_lnOR^2 / (pi^2/3)) [Weisburd et al. 2022, eq. 11.17]'
