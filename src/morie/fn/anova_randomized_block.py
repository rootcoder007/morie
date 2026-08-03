"""Block randomized ANOVA y = mu + alpha_j + beta_k + e.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["anova_randomized_block"]


def anova_randomized_block(y, treatment, block):
    """Block randomized ANOVA y = mu + alpha_j + beta_k + e

    Formula: y_ikj = mu + alpha_j + beta_k + e_ijk

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'f_treatment' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.9 eq.9.13
    """
    payload = dict(_ca_crim.anova_randomized_block(y, treatment, block))
    value = payload['f_treatment']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (9.13)"
    return RichResult(
        title='Block randomized ANOVA y = mu + alpha_j + beta_k + e',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca9e13: y_ikj = mu + alpha_j + beta_k + e_ijk [Weisburd et al. 2022, eq. 9.13]'
