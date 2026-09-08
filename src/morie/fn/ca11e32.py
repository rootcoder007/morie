"""se_r from se_d (unequal n).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_11_equation_32"]


def ca_chapter_11_equation_32(d, se_d, n1, n2):
    """se_r from se_d (unequal n)

    Formula: se_r = sqrt(((n1+n2)^2/(n1 n2)) se_d^2 / (se_d^2 + h)^3-form)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.32
    """
    value = _ca_crim.se_r_from_se_d(d, se_d, n1, n2)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.32)"
    return RichResult(
        title='se_r from se_d (unequal n)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e32: se_r = sqrt(((n1+n2)^2/(n1 n2)) se_d^2 / (se_d^2 + h)^3-form) [Weisburd et al. 2022, eq. 11.32]'
