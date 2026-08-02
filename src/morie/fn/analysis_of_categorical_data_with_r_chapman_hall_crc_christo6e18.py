"""GLMM with fixed slope g(mu) = b0 + b1 x + b0i.

Book-as-spec implementation; see reference for context.
"""

import numpy as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_18"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_18(b0, b1, x, random_intercept):
    """GLMM with fixed slope g(mu) = b0 + b1 x + b0i

    Formula: g(mu_ik) = b0 + b1 x_ik + b_0i

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (6.18).
    """
    value = _acd.glmm_linear_predictor(b0, b1, x, random_intercept)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (6.18)"
    return RichResult(
        title='GLMM with fixed slope g(mu) = b0 + b1 x + b0i',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '6e18: g(mu_ik) = b0 + b1 x_ik + b_0i [Bilder & Loughin 2025, eq. 6.18]'
