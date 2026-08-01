"""One-way ANOVA mean square between groups.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_9_equation_5"]


def ca_chapter_9_equation_5(groups):
    """One-way ANOVA mean square between groups

    Formula: MS_between = sum n_j (ybar_j - ybar)^2 / (a - 1)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'ms_between' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.9 eq.9.5
    """
    payload = dict(_ca_crim.anova_oneway(groups))
    value = payload['ms_between']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (9.5)"
    return RichResult(
        title='One-way ANOVA mean square between groups',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca9e5: MS_between = sum n_j (ybar_j - ybar)^2 / (a - 1) [Weisburd et al. 2022, eq. 9.5]'
