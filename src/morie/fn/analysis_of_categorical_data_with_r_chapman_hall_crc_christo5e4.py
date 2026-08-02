"""Variance of the model-averaged estimate.

Book-as-spec implementation; see reference for context.
"""

import numpy as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_5_equation_4"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_5_equation_4(taus, thetas, variances):
    """Variance of the model-averaged estimate

    Formula: Var = sum tau_m [(theta_m - theta_MA)^2 + Var(theta_m)]

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (5.4).
    """
    value = _acd.model_averaged_variance(taus, thetas, variances)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (5.4)"
    return RichResult(
        title='Variance of the model-averaged estimate',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '5e4: Var = sum tau_m [(theta_m - theta_MA)^2 + Var(theta_m)] [Bilder & Loughin 2025, eq. 5.4]'
