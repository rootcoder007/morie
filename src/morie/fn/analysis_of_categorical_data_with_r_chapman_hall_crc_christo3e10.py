"""Baseline-category logit for a contingency table.

Book-as-spec implementation; see reference for context.
"""

import numpy as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_10"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_10(bj0, bjs, xs):
    """Baseline-category logit for a contingency table

    Formula: log(pi_j/pi_1) = bj0 + bj2 x2 + ... + bjI xI

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (3.10).
    """
    value = _acd.baseline_logit(bj0, bjs, xs)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (3.10)"
    return RichResult(
        title='Baseline-category logit for a contingency table',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '3e10: log(pi_j/pi_1) = bj0 + bj2 x2 + ... + bjI xI [Bilder & Loughin 2025, eq. 3.10]'
