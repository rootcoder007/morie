"""-2 log(Lambda) in estimated-probability form.

Book-as-spec implementation; see reference for context.
"""

import numpy as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_7"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_7(loglik_null, loglik_full):
    """-2 log(Lambda) in estimated-probability form

    Formula: -2 sum y log(pi0/pia) + (1-y) log((1-pi0)/(1-pia))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (2.7).
    """
    value = _acd.lrt_statistic(loglik_null, loglik_full)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (2.7)"
    return RichResult(
        title='-2 log(Lambda) in estimated-probability form',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '2e7: -2 sum y log(pi0/pia) + (1-y) log((1-pi0)/(1-pia)) [Bilder & Loughin 2025, eq. 2.7]'
