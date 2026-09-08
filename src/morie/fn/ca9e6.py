"""One-way ANOVA mean square within groups.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_9_equation_6"]


def ca_chapter_9_equation_6(groups):
    """One-way ANOVA mean square within groups

    Formula: MS_within = sum sum (y_ij - ybar_j)^2 / (N - a)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'ms_within' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.9 eq.9.6
    """
    payload = dict(_ca_crim.anova_oneway(groups))
    value = payload['ms_within']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (9.6)"
    return RichResult(
        title='One-way ANOVA mean square within groups',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca9e6: MS_within = sum sum (y_ij - ybar_j)^2 / (N - a) [Weisburd et al. 2022, eq. 9.6]'
