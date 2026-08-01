"""Naive randomized experiment ANOVA y = mu + alpha_j + e.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_9_equation_12"]


def ca_chapter_9_equation_12(groups):
    """Naive randomized experiment ANOVA y = mu + alpha_j + e

    Formula: y_ij = mu + alpha_j + e_ij

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'f' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.9 eq.9.12
    """
    payload = dict(_ca_crim.anova_oneway(groups))
    value = payload['f']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (9.12)"
    return RichResult(
        title='Naive randomized experiment ANOVA y = mu + alpha_j + e',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca9e12: y_ij = mu + alpha_j + e_ij [Weisburd et al. 2022, eq. 9.12]'
