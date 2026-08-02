"""Jackknife variance of a survey proportion.

Book-as-spec implementation; see reference for context.
"""

import numpy as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_10"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_10(replicate_estimates, full_estimate):
    """Jackknife variance of a survey proportion

    Formula: Var(pi_i) = ((R-1)/R) sum_r (pi_i^(r) - pi_i)^2

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (6.10).
    """
    value = _acd.jackknife_variance(replicate_estimates, full_estimate)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (6.10)"
    return RichResult(
        title='Jackknife variance of a survey proportion',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '6e10: Var(pi_i) = ((R-1)/R) sum_r (pi_i^(r) - pi_i)^2 [Bilder & Loughin 2025, eq. 6.10]'
