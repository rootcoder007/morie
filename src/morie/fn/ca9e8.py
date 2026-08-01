"""Repeated measures MS_subjects.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_9_equation_8"]


def ca_chapter_9_equation_8(groups):
    """Repeated measures MS_subjects

    Formula: MS_subjects = b sum(ybar_ij - ybar_.j)^2 / (N - a)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'ms_subjects' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.9 eq.9.8
    """
    payload = dict(_ca_crim.repeated_measures_ms(groups))
    value = payload['ms_subjects']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (9.8)"
    return RichResult(
        title='Repeated measures MS_subjects',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca9e8: MS_subjects = b sum(ybar_ij - ybar_.j)^2 / (N - a) [Weisburd et al. 2022, eq. 9.8]'
