"""Repeated measures MS_B:subjects (B x subjects interaction).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_9_equation_9"]


def ca_chapter_9_equation_9(groups):
    """Repeated measures MS_B:subjects (B x subjects interaction)

    Formula: MS_Bsubjects = n_b sum(y_ijk - ybar_ij - ybar_k + ybar_.j)^2 / ((N-a)(b-1))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'ms_b_subjects' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.9 eq.9.9
    """
    payload = dict(_ca_crim.repeated_measures_ms(groups))
    value = payload['ms_b_subjects']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (9.9)"
    return RichResult(
        title='Repeated measures MS_B:subjects (B x subjects interaction)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca9e9: MS_Bsubjects = n_b sum(y_ijk - ybar_ij - ybar_k + ybar_.j)^2 / ((N-a)(b-1)) [Weisburd et al. 2022, eq. 9.9]'
