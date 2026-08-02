"""Likelihood ratio test for two binomial proportions.

Book-as-spec implementation; see reference for context.
"""

import numpy as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_equation_8"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_equation_8(w1, n1, w2, n2):
    """Likelihood ratio test for two binomial proportions

    Formula: -2 log(Lambda) = -2 sum_j [w_j log(pibar/pihat_j) + ...]

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'stat' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (1.8).
    """
    payload = dict(_acd.lrt_two_groups(w1, n1, w2, n2))
    value = float(payload['stat'])
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (1.8)"
    return RichResult(
        title='Likelihood ratio test for two binomial proportions',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '1e8: -2 log(Lambda) = -2 sum_j [w_j log(pibar/pihat_j) + ...] [Bilder & Loughin 2025, eq. 1.8]'
