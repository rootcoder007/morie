"""Variance of the linear predictor.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_16"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_16(xs, cov):
    """Variance of the linear predictor

    Formula: Var(b0 + b1 x1 + ... + bp xp) = sum_i sum_j x_i x_j Cov(b_i, b_j)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (2.16).
    """
    value = _acd.linear_predictor_variance(xs, cov)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (2.16)"
    return RichResult(
        title='Variance of the linear predictor',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '2e16: Var(b0 + b1 x1 + ... + bp xp) = sum_i sum_j x_i x_j Cov(b_i, b_j) [Bilder & Loughin 2025, eq. 2.16]'
