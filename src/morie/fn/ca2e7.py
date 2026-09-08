"""Two-IV regression coefficient for x1 from correlations.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_2_equation_7"]


def ca_chapter_2_equation_7(r_y1, r_y2, r_12, s_y, s_1, s_2):
    """Two-IV regression coefficient for x1 from correlations

    Formula: b_x1 = ((r_y1 - r_y2 r_12)/(1 - r_12^2)) (s_y/s_1)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'b1' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.2 eq.2.7
    """
    payload = dict(_ca_crim.ols_two_iv(r_y1, r_y2, r_12, s_y, s_1, s_2))
    value = payload['b1']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (2.7)"
    return RichResult(
        title='Two-IV regression coefficient for x1 from correlations',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca2e7: b_x1 = ((r_y1 - r_y2 r_12)/(1 - r_12^2)) (s_y/s_1) [Weisburd et al. 2022, eq. 2.7]'
