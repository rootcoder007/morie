"""Random-intercept GLMM linear predictor g(mu) = b0 + b0i.

Book-as-spec implementation; see reference for context.
"""

import numpy as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_17"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_17(b0, random_intercept):
    """Random-intercept GLMM linear predictor g(mu) = b0 + b0i

    Formula: g(mu_ik) = b0 + b_0i

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (6.17).
    """
    value = _acd.glmm_linear_predictor(b0, 0.0, 0.0, random_intercept)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (6.17)"
    return RichResult(
        title='Random-intercept GLMM linear predictor g(mu) = b0 + b0i',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '6e17: g(mu_ik) = b0 + b_0i [Bilder & Loughin 2025, eq. 6.17]'
