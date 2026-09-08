"""Variance components model y_ij = beta0 + u_j + e_ij.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_7_equation_3"]


def ca_chapter_7_equation_3(beta0, u_j, e_ij):
    """Variance components model y_ij = beta0 + u_j + e_ij

    Formula: y_ij = beta0 + u_j + e_ij (cluster treated as a random effect)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.7 eq.7.3
    """
    value = _ca_crim.multilevel_predict(beta0, [], [], [u_j], e_ij)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (7.3)"
    return RichResult(
        title='Variance components model y_ij = beta0 + u_j + e_ij',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca7e3: y_ij = beta0 + u_j + e_ij (cluster treated as a random effect) [Weisburd et al. 2022, eq. 7.3]'
