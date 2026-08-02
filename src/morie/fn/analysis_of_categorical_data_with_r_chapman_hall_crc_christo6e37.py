"""Spline odds ratio from basis differences.

Book-as-spec implementation; see reference for context.
"""

import numpy as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_37"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_37(betas, basis_fns, a, b_pt):
    """Spline odds ratio from basis differences

    Formula: exp(f(a) - f(b)) = exp(sum b_j (h_j(a) - h_j(b)))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (6.37).
    """
    value = _acd.spline_odds_ratio(betas, basis_fns, a, b_pt)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (6.37)"
    return RichResult(
        title='Spline odds ratio from basis differences',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '6e37: exp(f(a) - f(b)) = exp(sum b_j (h_j(a) - h_j(b))) [Bilder & Loughin 2025, eq. 6.37]'
