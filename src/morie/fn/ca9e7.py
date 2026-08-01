"""One-way ANOVA F = MS_between / MS_within.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_9_equation_7"]


def ca_chapter_9_equation_7(groups):
    """One-way ANOVA F = MS_between / MS_within

    Formula: F = MS_between / MS_within

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'f' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.9 eq.9.7
    """
    payload = dict(_ca_crim.anova_oneway(groups))
    value = payload['f']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (9.7)"
    return RichResult(
        title='One-way ANOVA F = MS_between / MS_within',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca9e7: F = MS_between / MS_within [Weisburd et al. 2022, eq. 9.7]'
