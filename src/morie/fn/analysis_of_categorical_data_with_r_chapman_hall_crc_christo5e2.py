"""BIC-based posterior model probabilities.

Book-as-spec implementation; see reference for context.
"""

import numpy as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_5_equation_2"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_5_equation_2(bics):
    """BIC-based posterior model probabilities

    Formula: tau_m = exp(-Delta_m/2)/sum_a exp(-Delta_a/2)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (5.2).
    """
    value = float(_acd.bic_posterior_probs(bics)[0])
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (5.2)"
    return RichResult(
        title='BIC-based posterior model probabilities',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '5e2: tau_m = exp(-Delta_m/2)/sum_a exp(-Delta_a/2) [Bilder & Loughin 2025, eq. 5.2]'
